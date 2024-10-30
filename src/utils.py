import os
import argparse
import random
from sklearn.model_selection import train_test_split
import pickle
import json
import pprint
from tqdm import tqdm
import dgl
from dgl.data.utils import load_graphs
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from dgl.dataloading import GraphDataLoader
from dgl import KHopGraph, save_graphs
import dgl.function as fn

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from functools import partial

from line_profiler import profile

NAME_MAP = {
    'n': "Node",
    'e': "Edge",
    'g': "Graph",
}

DATASETS = ['reddit', 'weibo', 'amazon', 'yelp', 'tfinance', 'elliptic', 'tolokers', 'questions', 'dgraphfin', 'tsocial', 'hetero/amazon', 'hetero/yelp', 
            'uni-tsocial', 
            'mnist/dgl/mnist0', 'mnist/dgl/mnist1', 
            'mutag/dgl/mutag0', 
            'bm/dgl/bm_mn_dgl', 'bm/dgl/bm_ms_dgl', 'bm/dgl/bm_mt_dgl',
            'tfinace'
            ]

EPS = 1e-12 # for nan
ROOT_SEED = 3407

def log_loss(tags:str, loss_item_dicts):
    for tag, loss_item_dict in zip(tags,loss_item_dicts):
        print(f"{tag} loss  ", end='')
        for k,v in loss_item_dict.items():
            print("  {}: {:.4f}".format(
                NAME_MAP[k],
                v
            ), end='')
        print("")



# ======================================================================
#   Model activation/normalization creation function
# ======================================================================

def obtain_act(name=None):
    """
    Return activation function module
    """
    if name == 'relu':
        act = nn.ReLU(inplace=True)
    elif name == "gelu":
        act = nn.GELU()
    elif name == "prelu":
        act = nn.PReLU()
    elif name == "elu":
        act = nn.ELU()
    elif name == "leakyrelu":
        act = nn.LeakyReLU()
    elif name == "tanh":
        act = nn.Tanh()
    elif name == "sigmoid":
        act = nn.Sigmoid()
    elif name is None:
        act = nn.Identity()
    else:
        raise NotImplementedError("{} is not implemented.".format(name))

    return act


def obtain_norm(name):
    """
    Return normalization function module
    """
    if name == "layernorm":
        norm = nn.LayerNorm
    elif name == "batchnorm":
        norm = nn.BatchNorm1d
    elif name == "instancenorm":
        norm = partial(nn.InstanceNorm1d, affine=True, track_running_stats=True)
    else:
        return nn.Identity

    return norm


def obtain_pooler(pooling):
    """
    Return pooling function module
    """
    if pooling == "mean":
        pooler = AvgPooling()
    elif pooling == "max":
        pooler = MaxPooling()
    elif pooling == "sum":
        pooler = SumPooling()
    else:
        raise NotImplementedError

    return pooler


# ======================================================================
#   Data augmentation funciton
# ======================================================================

def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng

# -------------------

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]

def collate_pretrain(graphs):
    # `samples` is a list of graphs
    batched_graph = dgl.batch(graphs) # Note that this could be very slow for multi-processing
    return batched_graph

def collate_mlp(samples):
    graphs, labels_dict_list = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels_dict = {}
    if labels_dict_list[0]['node_labels'] is not None:
        batched_labels_dict['node_labels'] = torch.cat([d['node_labels'] for d in labels_dict_list])
    if labels_dict_list[0]['edge_labels'] is not None:
        batched_labels_dict['edge_labels'] = torch.cat([d['edge_labels'] for d in labels_dict_list])
    if labels_dict_list[0]['graph_labels'] is not None:
        batched_labels_dict['graph_labels'] = torch.cat([d['graph_labels'].reshape(1) for d in labels_dict_list])
    return batched_graph, batched_labels_dict

def collate_with_sp(samples):
    graphs, labels_dict_list, khop_graphs = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_khop_graph = dgl.batch(khop_graphs)
    batched_labels_dict = {}
    if labels_dict_list[0]['node_labels'] is not None:
        batched_labels_dict['node_labels'] = torch.cat([d['node_labels'] for d in labels_dict_list])
    if labels_dict_list[0]['edge_labels'] is not None:
        batched_labels_dict['edge_labels'] = torch.cat([d['edge_labels'] for d in labels_dict_list])
    if labels_dict_list[0]['graph_labels'] is not None:
        batched_labels_dict['graph_labels'] = torch.cat([d['graph_labels'].reshape(1) for d in labels_dict_list])
    return batched_graph, batched_labels_dict, batched_khop_graph


def set_seed(seed=ROOT_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        torch.use_deterministic_algorithms(mode=True)

def select_topk_star_debug(xs, xs_ids, x0, x0_id):
    x0 = (x0, x0_id)
    xs = list(zip(xs, xs_ids))

    feature_id = 0

    up = 0
    down = torch.pow( x0[0][feature_id], 2 ) + EPS # element wise
    best = up / down
    greedy = lambda xi: - (x0[0][feature_id] - xi[0][feature_id])**2 / (xi[0][feature_id]**2)
    xs_sorted = sorted(xs, key=greedy)
    nbs = []
    for i,xi in enumerate(xs_sorted):
        tmp_up = (x0[0][feature_id] - xi[0][feature_id])**2
        tmp_down = xi[0][feature_id]**2 + EPS
        if best < tmp_up/tmp_down:
            up += tmp_up
            down += tmp_down
            best = up / down
            nbs.append(xi[1]) # sotre the id
        else:
            break
    return 

def select_all_khop(star_khop_graph_big, central_node_id, khop, select_topk):
    pres = star_khop_graph_big.predecessors(central_node_id) # in edges
    sucs = star_khop_graph_big.successors(central_node_id) # out edges
    node_ids = torch.unique(torch.cat([pres, sucs], dim=0))
    nbs = torch.unique(node_ids)
    weights = torch.ones(nbs.shape[0], 1)
    weights[:-1, 0] = weights[:-1, 0]/(nbs.shape[0] + EPS)
    return nbs, weights

def select_rand_khop(star_khop_graph_big, central_node_id, khop, select_topk):
    pres = star_khop_graph_big.predecessors(central_node_id) # in edges
    sucs = star_khop_graph_big.successors(central_node_id) # out edges
    node_ids = torch.unique(torch.cat([pres, sucs], dim=0))
    idx = torch.randperm(node_ids.shape[0])
    nbs = node_ids[idx[:100]]
    weights = torch.ones(nbs.shape[0], 1)
    weights[:-1, 0] = weights[:-1, 0]/(nbs.shape[0] + EPS)
    return nbs, weights

def select_topk_star_normft(star_khop_graph_big, node_ids, central_node_id):
    h_xs, id_xs, h_x0, id_x0 = star_khop_graph_big.ndata['feature_normed'][node_ids], node_ids, star_khop_graph_big.ndata['feature_normed'][central_node_id], central_node_id

    xs = list(zip(h_xs, id_xs))
    x0 = (h_x0, id_x0)

    up = 0
    down = torch.pow( x0[0], 2 ) + EPS # element wise
    best = up / down
    greedy = lambda xi: - (x0[0] - xi[0])**2 / (xi[0]**2)
    xs_sorted = sorted(xs, key=greedy)
    nbs = []
    for i,xi in enumerate(xs_sorted):
        tmp_up = (x0[0] - xi[0])**2
        tmp_down = xi[0]**2 + EPS
        if best < tmp_up/tmp_down:
            up += tmp_up
            down += tmp_down
            best = up / down
            nbs.append(xi[1]) # sotre the id
        else:
            break
    return nbs

def select_topk_star_unionft(star_khop_graph_big, node_ids, central_node_id):
    h_xs, id_xs, h_x0, id_x0 = star_khop_graph_big.ndata['feature'][node_ids], node_ids, star_khop_graph_big.ndata['feature'][central_node_id], central_node_id


    nbs = set()
    for feature_id in range(h_xs.shape[1]):
        xs = list(zip(h_xs[:, feature_id], id_xs))
        x0 = (h_x0[feature_id], id_x0)

        up = 0
        down = torch.pow( x0[0], 2 ) # element wise
        best = up / down
        greedy = lambda xi: - (x0[0] - xi[0])**2 / (xi[0]**2)
        xs_sorted = sorted(xs, key=greedy)

        for i,xi in enumerate(xs_sorted):
            tmp_up = (x0[0] - xi[0])**2
            tmp_down = xi[0]**2
            if best < tmp_up/tmp_down:
                up += tmp_up
                down += tmp_down
                best = up / down
                nbs.add(xi[1]) # sotre the id
    return list(nbs)

def get_star_topk_nbs(star_khop_graph_big, central_node_id, khop, select_topk):
    star_khop_graph_in = star_khop_graph_big.sample_neighbors([central_node_id],fanout=-1, edge_dir='in')
    star_khop_graph_out = star_khop_graph_big.sample_neighbors([central_node_id],fanout=-1, edge_dir='out')

    node_ids_in = star_khop_graph_in.edges()[0]
    node_ids_out = star_khop_graph_out.edges()[1]
    node_ids = torch.cat([node_ids_in, node_ids_out], dim=0)
    node_ids = torch.unique(node_ids)


    nbs = select_topk(star_khop_graph_big, node_ids, central_node_id)
    nbs.append(torch.tensor(central_node_id).long()) # make sure self is added
    weights = torch.ones(len(nbs), 1)*0.5
    weights[:-1, 0] = weights[:-1, 0]/(len(nbs) + EPS)

    return nbs, weights

@profile
def get_convtree_topk_nbs_norm(graph_whole, xi, khop, select_topk):
    '''
        return topk neighbors weight matrix in Conv Tree graph setting
    '''
    # find all 1st-order neighbours
    pres = graph_whole.predecessors(xi) # in edges
    sucs = graph_whole.successors(xi) # out edges
    nbs_xi = torch.unique(torch.cat([pres, sucs], dim=0)) # FIXME: if all bidirected, delete this for performance
    if nbs_xi.shape[0] == 0:
        # no neighbours
        return tuple([xi]), tuple([1.0])
    # some refrences for help
    xf = graph_whole.ndata['feature_normed']
    Pij = {} 
    Pik = {}
    Pij_tmp = {}
    Smaxj_list = []
    quant = lambda x: - x[1] / x[2]
    for xj in nbs_xi:
        # clear tmp ik for j
        Pik_tmp = {}
        # add parent edge
        aj = ( xf[xj] - xf[xi] )**2
        bj = ( xf[xj] )**2
        Smaxj = aj / bj
        # get xj's neighbours 
        pres = graph_whole.predecessors(xj) # in edges
        sucs = graph_whole.successors(xj) # out edges
        nbs_xj = torch.unique(torch.cat([pres, sucs], dim=0))
        if nbs_xj.shape[0] == 0:
            # xj no neighbours
            Pij_tmp[xj] = 0.5 # 1/2
        else:
            Pij_tmp[xj] = 0.25 # 1/4, because it has to avg with sons
            num_hop2 = 0 # how many sons has been selected, could be 0?
            ss = [ (xk.item(), (xf[xk]-xf[xj])**2, xf[xk]**2) for xk in nbs_xj] # store in (k, ak,bk) form
            ss.sort(key = lambda x: -x[1]/x[2]) # from big to small ak/bk
            # loop to find the optimal value
            for xk, ak, bk in ss:
                if ak / bk > Smaxj:
                    num_hop2 += 1
                    # update the best sons
                    aj += ak
                    bj += bk
                    Smaxj = aj / bj
                    Pik_tmp[xk] = 0.25 # Pik_tmp[xk] = 1/4
                else:
                    # the rest is impossible to make the ans bigger
                    break
            if num_hop2 != 0:
                # update all Pik_tmp
                for xk in Pik_tmp:
                    Pik_tmp[xk] /= num_hop2
                    # add to global Pik
                    if xk in Pik:
                        Pik[xk] += Pik_tmp[xk]
                    else:
                        Pik[xk] = Pik_tmp[xk]
            else:
                Pij_tmp[xj] = 0.5
        Smaxj_list.append((xj, aj, bj)) # j, aj, bj
    Smaxj_list = sorted(Smaxj_list, key=lambda x: -x[1]/x[2]) # from big to small
    ai = Smaxj_list[0][1]
    bi = Smaxj_list[0][2]
    RQ_max = ai / bi # at least the largest one should be selected
    num_hop1 = 1
    for xj, aj, bj in Smaxj_list[1:]: # check the rest
        if aj / bj > RQ_max:
            num_hop1 += 1
            ai += aj
            bi += bj
            RQ_max = ai / bi
            Pij[xj] = Pij_tmp[xj] # select xj
        else:
            break
    
    for xj in Pij:
        # update all Pij
        Pij[xj] /= num_hop1
    Pij[xi] = 0.5 # self loop

    Pfinal = {k:v for k,v in Pij.items()}
    for k,v in Pik.items():
        if k in Pfinal:
            Pfinal[k] += v
        else:
            Pfinal[k] = v

    adj_list, weight_list  = tuple(Pfinal.keys()), tuple(Pfinal.values())

    return adj_list, weight_list

class Dataset:
    def __init__(self, name='tfinance', prefix='../datasets/', labels_have="ng", sp_type='star+norm', debugnum = -1):
        self.full_name = prefix + name
        ### avoid repeat calcs
        self.prepare_dataset_done = False
        self.make_sp_matrix_graph_list_done = False

        if "unified" not in prefix and "edge_labels" not in prefix:
            graph = load_graphs(prefix + name)[0][0]
            self.name = name
            self.graph = graph
            self.in_dim = graph.ndata['feature'].shape[1]
        else:
            print("Unified dataset ", prefix + name, labels_have)
            self.labels_have = labels_have
            # graph list as well as node labels
            graph, label = load_graphs(prefix + name)
            self.name = name
            if debugnum == -1:
                self.graph_list = graph
            else:
                self.graph_list = graph[:debugnum]
            
            if 'g' in self.labels_have:
                self.graph_label = label['glabel'][:len(self.graph_list)]
            else:
                self.graph_label = [None for _ in range(len(self.graph_list))]
            
            self.in_dim = self.graph_list[0].ndata['feature'].shape[1]
            self.sp_type = sp_type
            self.sp_method, self.agg_ft = sp_type.split('+')

            if self.sp_method == 'star':
                self.get_sp_adj_list = get_star_topk_nbs
                if self.agg_ft == 'norm':
                    self.select_topk_fn = select_topk_star_normft
                elif self.agg_ft == "union":
                    self.select_topk_fn = select_topk_star_unionft
                else:
                    raise NotImplementedError
            elif self.sp_method == 'convtree':
                if self.agg_ft == 'norm':
                    self.get_sp_adj_list = get_convtree_topk_nbs_norm
                    self.select_topk_fn = None
                elif self.agg_ft == "union":
                    raise NotImplementedError
                else:
                    raise NotImplementedError
            elif self.sp_method == 'khop':
                self.get_sp_adj_list = select_all_khop
                self.select_topk_fn = None
            elif self.sp_method == 'rand':
                self.get_sp_adj_list = select_rand_khop
                self.select_topk_fn = None
            else:
                raise NotImplementedError
            

    def make_sp_matrix_graph_list(self, khop=1, sp_type='star+union', load_kg = False):
        if self.make_sp_matrix_graph_list_done:
            return
        # khop graph list
        self.sp_matrix_graph_list = []
        self.sp_matrix_graphs_filename = f"{self.full_name}.khop_{khop}.sp_type_{self.sp_type}.sp_matrix"

        if load_kg and os.path.exists(self.sp_matrix_graphs_filename):
            self.sp_matrix_graph_list, _ = load_graphs(self.sp_matrix_graphs_filename)
        else:
            for idx,graph in enumerate(tqdm(self.graph_list)):
                with graph.local_scope():
                    if self.agg_ft == 'norm':
                        if self.full_name.endswith("mutag0"):
                            graph.ndata['feature_normed'] =  graph.ndata['feature'].argmax(dim=1)
                        else:
                            graph.ndata['feature_normed'] =  graph.ndata['feature']
                            # norm it
                            graph.ndata['feature_normed'] -= graph.ndata['feature_normed'].min(0, keepdim=True)[0]
                            graph.ndata['feature_normed'] /= graph.ndata['feature_normed'].max(0, keepdim=True)[0] + EPS
                            graph.ndata['feature_normed'] = torch.norm(graph.ndata['feature_normed'], dim=1)
                    if khop !=0 :
                        sp_matrix_graph = dgl.graph(([], []))
                        sp_matrix_graph.add_nodes(graph.num_nodes()) # keep the node num same
                        if self.sp_method == 'star':
                            assert khop == 1
                            transform = KHopGraph(khop)
                            tmp_graph = transform(graph)
                            tmp_graph = tmp_graph.to_simple()
                            tmp_graph = tmp_graph.remove_self_loop()
                        elif self.sp_method == 'convtree':
                            assert khop == 2
                            # we directly use the big graph
                            tmp_graph= graph
                        elif self.sp_method == 'khop' or self.sp_method == 'rand':
                            transform = KHopGraph(khop)
                            tmp_graph = transform(graph)
                            tmp_graph = tmp_graph.to_simple()
                            tmp_graph = tmp_graph.remove_self_loop()
                        for central_node_id in graph.nodes():
                            adj_list, weight_list = self.get_sp_adj_list(tmp_graph, central_node_id.item(), khop, self.select_topk_fn)
                            sp_matrix_graph.add_edges(adj_list, central_node_id.long(), {'pw': torch.tensor(weight_list) }) # adj_list->node_id, edata['pw'] = weights
                        self.sp_matrix_graph_list.append(sp_matrix_graph)
                    else:
                        self.sp_matrix_graph_list.append(dgl.graph(([], []))) # make a empty graph
                    if self.agg_ft == 'norm':
                        graph.ndata.pop('feature_normed') # remove normed feature
                if self.is_single_graph:
                    break # we only need 1 khop graph for single graph datasets
            
            save_graphs(self.sp_matrix_graphs_filename, self.sp_matrix_graph_list)
        
        if khop != 0:
            # fix nan
            for kg in self.sp_matrix_graph_list:
                kg.edata['pw'] = torch.nan_to_num(kg.edata['pw'])

        self.make_sp_matrix_graph_list_done = True
        return

    def prepare_dataset(self, total_trials=1):
        '''
            prepare the multi trials dataset and make subpooling matrix
        '''
        if self.prepare_dataset_done:
            return
        # node level stuff
        self.node_label = []
        self.edge_label = []
        self.node_train_masks = []
        self.node_val_masks = []
        self.node_test_masks = []
        
        # some preprocess
        for idx,graph in enumerate(tqdm(self.graph_list)):
            graph.ndata['feature'] = graph.ndata['feature'].float()
            if 'n' in self.labels_have:
                self.node_label.append(graph.ndata['node_label'])
            if 'e' in self.labels_have:
                self.edge_label.append(graph.edata['edge_label'])
        
        # graph level split
        train_ratio, val_ratio = [0.4, 0.2] # default ratio
        if self.name in ['tolokers', 'questions']:
            train_ratio, val_ratio = 0.5, 0.25
        if self.name in ['uni-tsocial', 'tsocial', 'tfinance', 'reddit', 'weibo']:
            train_ratio, val_ratio = 0.4, 0.2
        if self.name in ['amazon', 'yelp']:
            train_ratio, val_ratio = 0.7, 0.1
        if "mnist0" in self.name or "mnist1" in self.name:
            train_ratio, val_ratio = 0.1, 0.1
        samples = total_trials
        if len(self.graph_list) > 1: # multi-graph
            self.is_single_graph = False
            graph_num = len(self.graph_list)
            indexs = list(range(graph_num))
            self.graph_train_masks = torch.zeros([graph_num, samples]).bool()
            self.graph_val_masks = torch.zeros([graph_num, samples]).bool()
            self.graph_test_masks = torch.zeros([graph_num, samples]).bool()
            for i in tqdm(range(samples)):
                seed = ROOT_SEED+samples*i
                set_seed(seed)
                idx_train, idx_rest, y_train, y_rest = train_test_split(indexs, self.graph_label, stratify=self.graph_label, train_size=train_ratio, random_state=seed, shuffle=True)
                idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, train_size=val_ratio/(1-train_ratio), random_state=seed, shuffle=True)
                self.graph_train_masks[idx_train,i] = 1
                self.graph_val_masks[idx_valid,i] = 1
                self.graph_test_masks[idx_test,i] = 1
        else:# single graph
            self.is_single_graph = True
            ori_graph = self.graph_list[0]
            self.graph_list = []
            graph_num = 3 * samples
            node_indexs = list(range(ori_graph.num_nodes()))
            node_labels = self.node_label[0]
            self.graph_train_masks = torch.zeros([graph_num, samples]).bool()
            self.graph_val_masks = torch.zeros([graph_num, samples]).bool()
            self.graph_test_masks = torch.zeros([graph_num, samples]).bool()

            
            self.train_mask_node = []
            self.val_mask_node = []
            self.test_mask_node = []

            self.train_mask_edge = [] # original edge ids
            self.val_mask_edge = []
            self.test_mask_edge = []
            for i in tqdm(range(samples)):
                seed = ROOT_SEED+samples*i
                set_seed(seed)
                # generate 3 subset by masking
                idx_train, idx_rest, y_train, y_rest = train_test_split(node_indexs, node_labels, stratify=node_labels, train_size=train_ratio, random_state=seed, shuffle=True)
                idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, train_size=val_ratio/(1-train_ratio), random_state=seed, shuffle=True)
                # setup the node masks
                self.train_mask_node.append( torch.tensor(idx_train).long() )
                self.val_mask_node.append( torch.tensor(idx_valid).long() )
                self.test_mask_node.append( torch.tensor(idx_test).long() )
                # setup the edge masks
                train_graph_tmp = dgl.node_subgraph(ori_graph, idx_train, store_ids=True)
                val_graph_tmp = dgl.node_subgraph(ori_graph, idx_valid, store_ids=True)
                test_graph_tmp = dgl.node_subgraph(ori_graph, idx_test, store_ids=True)
                self.train_mask_edge.append( train_graph_tmp.edata[dgl.EID] ) # original edge ids
                self.val_mask_edge.append( val_graph_tmp.edata[dgl.EID] )
                self.test_mask_edge.append( test_graph_tmp.edata[dgl.EID] )
                
                self.graph_list.append(ori_graph)
                self.graph_list.append(ori_graph)
                self.graph_list.append(ori_graph)
                self.graph_train_masks[i*3,i] = 1
                self.graph_val_masks[i*3+1,i] = 1
                self.graph_test_masks[i*3+2,i] = 1

        self.prepare_dataset_done = True
        return

    def calc_embeddings(self, pretrain_model, device='cuda'):
        '''
            apply the pretrain_model to each graph and store their embeddings in .ndata['embeds']
            the original .ndata['feature'] will be removed to reduce GPU mem usage
        '''
        for idx,graph in enumerate(tqdm(self.graph_list)):
            graph = graph.to(device)
            graph.ndata['embeds'] = pretrain_model.embed(graph, graph.ndata['feature']).detach()
            graph.ndata.pop('feature') # remove the original features
            self.graph_list[idx] = graph.to('cpu') # save GPU mem

        return

    def split(self, trial_id=0):        
        '''
            split data for tranning final MLP predictor
        '''
        if not self.is_single_graph:
            # train sets, unit: graph
            self.train_labels_dict_list = []
            self.train_graphs = []
            self.train_sp_matrix_graphs = []
            for x in self.graph_train_masks[:, trial_id].nonzero().reshape(-1):
                self.train_graphs.append(self.graph_list[x])
                train_labels_dict = {
                    'node_labels': None,
                    'edge_labels': None,
                    'graph_labels': None,
                }
                if 'n' in self.labels_have:
                    train_labels_dict['node_labels'] = self.graph_list[x].ndata['node_label']
                if 'e' in self.labels_have:
                    train_labels_dict['edge_labels'] = self.graph_list[x].edata['edge_label']
                if 'g' in self.labels_have:
                    train_labels_dict['graph_labels'] = self.graph_label[x]
                self.train_labels_dict_list.append(train_labels_dict)
                self.train_sp_matrix_graphs.append(self.sp_matrix_graph_list[x])

            # val sets, unit: graph
            self.val_labels_dict_list = []
            self.val_graphs = []
            self.val_sp_matrix_graphs = []
            for x in self.graph_val_masks[:, trial_id].nonzero().reshape(-1):
                self.val_graphs.append(self.graph_list[x])
                val_labels_dict = {
                    'node_labels': None,
                    'edge_labels': None,
                    'graph_labels': None,
                }
                if 'n' in self.labels_have:
                    val_labels_dict['node_labels'] = self.graph_list[x].ndata['node_label']
                if 'e' in self.labels_have:
                    val_labels_dict['edge_labels'] = self.graph_list[x].edata['edge_label']
                if 'g' in self.labels_have:
                    val_labels_dict['graph_labels'] = self.graph_label[x]
                self.val_labels_dict_list.append(val_labels_dict)
                self.val_sp_matrix_graphs.append(self.sp_matrix_graph_list[x])

            # test sets, unit: graph
            self.test_labels_dict_list = []
            self.test_graphs = []
            self.test_sp_matrix_graphs = []
            for x in self.graph_test_masks[:, trial_id].nonzero().reshape(-1):
                self.test_graphs.append(self.graph_list[x])
                test_labels_dict = {
                    'node_labels': None,
                    'edge_labels': None,
                    'graph_labels': None,
                }
                if 'n' in self.labels_have:
                    test_labels_dict['node_labels'] = self.graph_list[x].ndata['node_label']
                if 'e' in self.labels_have:
                    test_labels_dict['edge_labels'] = self.graph_list[x].edata['edge_label']
                if 'g' in self.labels_have:
                    test_labels_dict['graph_labels'] = self.graph_label[x]
                self.test_labels_dict_list.append(test_labels_dict)
                self.test_sp_matrix_graphs.append(self.sp_matrix_graph_list[x])
        else:
            # train sets, unit: graph
            self.train_mask_node_cur = self.train_mask_node[trial_id]
            self.train_mask_edge_cur = self.train_mask_edge[trial_id]
            self.train_labels_dict_list = []
            self.train_graphs = []
            self.train_sp_matrix_graphs = []
            for x in self.graph_train_masks[:, trial_id].nonzero().reshape(-1):
                self.train_graphs.append(self.graph_list[x])
                train_labels_dict = {
                    'node_labels': None,
                    'edge_labels': None,
                    'graph_labels': None,
                }
                if 'n' in self.labels_have:
                    train_labels_dict['node_labels'] = self.graph_list[x].ndata['node_label'][self.train_mask_node_cur]
                if 'e' in self.labels_have:
                    train_labels_dict['edge_labels'] = self.graph_list[x].edata['edge_label'][self.train_mask_edge_cur]
                if 'g' in self.labels_have:
                    raise NotImplementedError
                    train_labels_dict['graph_labels'] = self.graph_label[x]
                self.train_labels_dict_list.append(train_labels_dict)
                self.train_sp_matrix_graphs.append(self.sp_matrix_graph_list[0])


            # val sets, unit: graph
            self.val_mask_node_cur = self.val_mask_node[trial_id]
            self.val_mask_edge_cur = self.val_mask_edge[trial_id]
            self.val_labels_dict_list = []
            self.val_graphs = []
            self.val_sp_matrix_graphs = []
            for x in self.graph_val_masks[:, trial_id].nonzero().reshape(-1):
                self.val_graphs.append(self.graph_list[x])
                val_labels_dict = {
                    'node_labels': None,
                    'edge_labels': None,
                    'graph_labels': None,
                }
                if 'n' in self.labels_have:
                    val_labels_dict['node_labels'] = self.graph_list[x].ndata['node_label'][self.val_mask_node_cur]
                if 'e' in self.labels_have:
                    val_labels_dict['edge_labels'] = self.graph_list[x].edata['edge_label'][self.val_mask_edge_cur]
                if 'g' in self.labels_have:
                    raise NotImplementedError
                    val_labels_dict['graph_labels'] = self.graph_label[x]
                self.val_labels_dict_list.append(val_labels_dict)
                self.val_sp_matrix_graphs.append(self.sp_matrix_graph_list[0])


            # test sets, unit: graph
            self.test_mask_node_cur = self.test_mask_node[trial_id]
            self.test_mask_edge_cur = self.test_mask_edge[trial_id]
            self.test_labels_dict_list = []
            self.test_graphs = []
            self.test_sp_matrix_graphs = []
            for x in self.graph_test_masks[:, trial_id].nonzero().reshape(-1):
                self.test_graphs.append(self.graph_list[x])
                test_labels_dict = {
                    'node_labels': None,
                    'edge_labels': None,
                    'graph_labels': None,
                }
                if 'n' in self.labels_have:
                    test_labels_dict['node_labels'] = self.graph_list[x].ndata['node_label'][self.test_mask_node_cur]
                if 'e' in self.labels_have:
                    test_labels_dict['edge_labels'] = self.graph_list[x].edata['edge_label'][self.test_mask_edge_cur]
                if 'g' in self.labels_have:
                    raise NotImplementedError
                    test_labels_dict['graph_labels'] = self.graph_label[x]
                self.test_labels_dict_list.append(test_labels_dict)
                self.test_sp_matrix_graphs.append(self.sp_matrix_graph_list[0])

    def get_graph_dataloaders(self, batch_size, trial_id=0):
        '''
            get dataloaders for the MLP
        '''
        self.split(trial_id=trial_id)
        # train loader
        train_graphs_with_all_labels = list(zip(
            self.train_graphs, 
            self.train_labels_dict_list
        ))
        train_dataloader = GraphDataLoader(
            train_graphs_with_all_labels, 
            batch_size=batch_size, 
            collate_fn=collate_mlp,
            shuffle=True
        )
        # val loader
        val_graphs_with_all_labels = list(zip(
            self.val_graphs, 
            self.val_labels_dict_list
        ))
        val_dataloader = GraphDataLoader(
            val_graphs_with_all_labels, 
            batch_size=batch_size, 
            collate_fn=collate_mlp,
            shuffle=True
        )
        # test loader
        test_graphs_with_all_labels = list(zip(
            self.test_graphs, 
            self.test_labels_dict_list
        ))
        test_dataloader = GraphDataLoader(
            test_graphs_with_all_labels, 
            batch_size=batch_size, 
            collate_fn=collate_mlp,
            shuffle=True
        )
        
        return train_dataloader, val_dataloader, test_dataloader

    def get_graph_and_sp_dataloaders(self, batch_size, trial_id=0):
        '''
            get dataloaders of graph and sampling matrix for the MLP
        '''
        self.split(trial_id=trial_id)
        # train loader
        train_graphs_with_all_labels = list(zip(
            self.train_graphs, 
            self.train_labels_dict_list,
            self.train_sp_matrix_graphs
        ))
        train_dataloader = GraphDataLoader(
            train_graphs_with_all_labels, 
            batch_size=batch_size, 
            collate_fn=collate_with_sp,
            shuffle=True
        )
        # val loader
        val_graphs_with_all_labels = list(zip(
            self.val_graphs, 
            self.val_labels_dict_list,
            self.val_sp_matrix_graphs
        ))
        val_dataloader = GraphDataLoader(
            val_graphs_with_all_labels, 
            batch_size=batch_size, 
            collate_fn=collate_with_sp,
            shuffle=True
        )
        # test loader
        test_graphs_with_all_labels = list(zip(
            self.test_graphs, 
            self.test_labels_dict_list,
            self.test_sp_matrix_graphs
        ))
        test_dataloader = GraphDataLoader(
            test_graphs_with_all_labels, 
            batch_size=batch_size, 
            collate_fn=collate_with_sp,
            shuffle=True
        )
        
        return train_dataloader, val_dataloader, test_dataloader

    def get_pretrain_dataloaders(self, batch_size):
        # pre-train loader, all graphs
        pretrain_dataloader = GraphDataLoader(
            self.graph_list, 
            batch_size=batch_size, 
            collate_fn=collate_pretrain,
            shuffle=True
        )
        return pretrain_dataloader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default="", help='addtional tags for distinguish result')
    parser.add_argument('--khop', type=int, default=0)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr_ft', type=float, default=0.003)
    parser.add_argument("--l2_ft", type=float, default=0)
    parser.add_argument('--epoch_ft', type=int, default=200)
    parser.add_argument("--stitch_mlp_layers", type=int, default=1, help="Number of hidden layer in stitch MLP")
    parser.add_argument("--final_mlp_layers", type=int, default=2, help="Number of hidden layer in final MLP")
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--metric', type=str, default='AUROC')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--task_level', type=str, default='unify')
    parser.add_argument('--node_loss_weight', type=float, default=1)
    parser.add_argument('--edge_loss_weight', type=float, default=1)
    parser.add_argument('--graph_loss_weight', type=float, default=1)
    parser.add_argument('--cross_modes', type=str, default="ng2ng")
    parser.add_argument('--sp_type', type=str, default='star+union', help="neighbor sampling strategy")
    parser.add_argument('--force_remake_sp',  action="store_true", help="force remaking neighbor sampling matrix")
    # pretrain model parameters
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument('--epoch_pretrain', type=int, default=100)
    parser.add_argument('--pretrain_model', type=str, default='graphmae')
    parser.add_argument('--kernels', type=str, default='gcn', help="Encoder/Decode GNN model types")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0, help="Coefficient of L2 penalty")
    parser.add_argument("--decay_rate", type=float, default=1, help="Decay rate of learning rate")
    parser.add_argument("--decay_step", type=int, default=100, help="Decay step of learning rate")
    parser.add_argument('--drop_rate', type=float, default=0)
    parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dimension")
    parser.add_argument("--num_layer_pretrain", type=int, default=2, help="Number of hidden layer in pretrain model")
    parser.add_argument("--act", type=str, default='leakyrelu', help="Activation function type")
    parser.add_argument("--act_ft", type=str, default='ReLU', help="Activation function for mlp")
    parser.add_argument("--norm", type=str, default="", help="Normlaization layer type")
    parser.add_argument("--concat", action="store_true", default=False, help="Indicator of where using raw and generated embeddings")
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    # GraphMAE
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Masking ratio for GraphMAE")
    parser.add_argument("--replace_ratio", type=float, default=0, help="Replace ratio for GraphMAE")
    
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate for node in training")
    
    
    args = parser.parse_args()
    return args


def save_results(results, save_file_name=None):
    save_file_name = save_file_name.replace('/', '')
    if not os.path.exists('../results/'):
        os.mkdir('../results/')
    results.transpose().to_excel('../results/{}.xlsx'.format(save_file_name))
    print('save to file: {}'.format(save_file_name))
