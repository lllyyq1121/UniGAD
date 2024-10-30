from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from utils import *
from predictors import *
from Pareto_fn import pareto_fn
from pcgrad_fn import pcgrad_fn

# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        # preds[probs[:,1] > thres] = 1
        preds[probs > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

LABEL_DICT_KEYS = {
    'n':"node_labels",
    'e':'edge_labels',
    'g':'graph_labels',
}

class UnifyMLPDetector(object):
    def __init__(self, pretrain_model, dataset, dataloaders, cross_mode, args):
        self.args = args

        self.train_dataloader = dataloaders[0]
        self.val_dataloader = dataloaders[1]
        self.test_dataloader = dataloaders[2]

        # the loss route
        input_route, output_route = cross_mode.split('2')
        self.input_route = [c for c in input_route] # ['n', 'e', 'g']
        self.output_route = [c for c in output_route] # ['n', 'e', 'g'] # the output of the model

        self.model = UNIMLP_E2E(
            in_feats=pretrain_model.in_dim,
            embed_dims=pretrain_model.embed_dim,
            khop=args.khop,
            activation=args.act_ft,
            graph_batch_num=args.batch_size,
            stitch_mlp_layers=args.stitch_mlp_layers,
            final_mlp_layers=args.final_mlp_layers,
            pretrain_model=pretrain_model,
            output_route=output_route,
            input_route=input_route,
            dropout_rate=args.dropout
        ).to(args.device)

        self.loss_weight_dict = {}
        if 'n' in self.output_route:
            node_ab_count, node_total_count = sum([x.sum() for x in dataset.node_label]), sum(x.shape[0] for x in dataset.node_label)
            self.loss_weight_dict['n'] = (
                1/(node_ab_count/node_total_count),
                args.node_loss_weight 
            )
        if 'e' in self.output_route:
            edge_ab_count, edge_total_count = sum([x.sum() for x in dataset.edge_label]), sum(x.shape[0] for x in dataset.edge_label)
            self.loss_weight_dict['e'] = (
                1/(edge_ab_count/edge_total_count),
                args.edge_loss_weight 
            )
        if 'g' in self.output_route:
            graph_ab_count, graph_total_count = dataset.graph_label.sum(), dataset.graph_label.shape[0]
            self.loss_weight_dict['g'] = (
                1/(graph_ab_count/graph_total_count),
                args.graph_loss_weight 
            )

        # masks for single graph
        if dataset.is_single_graph:
            mask_dicts = {}
            self.is_single_graph = True
            if 'n' in self.output_route:
                mask_dicts['n'] = {
                    'train': dataset.train_mask_node_cur,
                    'val': dataset.val_mask_node_cur,
                    'test': dataset.test_mask_node_cur
                }
            if 'e' in self.output_route:
                mask_dicts['e'] = {
                    'train': dataset.train_mask_edge_cur,
                    'val': dataset.val_mask_edge_cur,
                    'test': dataset.test_mask_edge_cur
                }
            if 'g' in self.output_route:
                # single graph cannot be classified
                raise NotImplementedError
            
            self.model.mask_dicts = mask_dicts
            self.model.single_graph = True

        self.best_score = -1
        self.patience_knt = 0
        


    def get_loss(self, logits_dict={}, labels_dict={}):
        loss_items_dict = {'n': 0, 'e': 0, 'g': 0}
        loss = None

        loss_list = []
        w_list = []
        c_list = []

        for o_r in logits_dict:
            partial_loss = F.cross_entropy(logits_dict[o_r], labels_dict[LABEL_DICT_KEYS[o_r]], weight=torch.tensor([1., self.loss_weight_dict[o_r][0]], device=self.args.device))
            if o_r in self.input_route:
                # loss = partial_loss if loss is None else (loss + partial_loss * self.loss_weight_dict[o_r][1])
                loss_list.append(partial_loss)
                w_list.append(1.0/len(self.input_route)) # FIXME: default loss average
                c_list.append(0.01)
            loss_items_dict[o_r] = partial_loss.item()

        # return loss_list, loss_items_dict
        
        new_w_list = pareto_fn(w_list, c_list, model=self.model, num_tasks=len(loss_list), loss_list=loss_list)
        loss = 0
        for i in range(len(w_list)):
            loss += new_w_list[i]*loss_list[i]
        
        return loss, loss_items_dict
    
    @torch.no_grad()
    def get_probs(self, logits_dict={}):
        probs_dict = {}
        for o_r in logits_dict:
            probs_dict[o_r] = logits_dict[o_r].softmax(1)[:, 1]
        return probs_dict

    @torch.no_grad()
    def _single_eval(self, labels, probs):
        score = {}
        with torch.no_grad():
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            if torch.is_tensor(probs):
                probs = probs.cpu().numpy()
            score['MacroF1'] = get_best_f1(labels, probs)[0]
            score['AUROC'] = roc_auc_score(labels, probs)
            score['AUPRC'] = average_precision_score(labels, probs)

        return score
    
    @torch.no_grad()
    def eval(self, labels_dict, probs_dict):
        result = {}
        for k in self.output_route:
            result[k] = self._single_eval(labels_dict[k], probs_dict[k])
        return result

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr_ft, weight_decay=self.args.l2_ft)
        for epoch in tqdm( range(self.args.epoch_ft) ):
            loss_items_total_train = {k:0 for k in self.output_route }
            total_loss_graph = 0
            total_loss_node = 0
            for batched_data in self.train_dataloader:
                batched_graph, batched_labels_dict, batched_khop_graph = batched_data
                # FIXME: device issue?
                batched_graph = batched_graph.to(self.args.device)
                for k,v in batched_labels_dict.items():
                    batched_labels_dict[k] = v.to(self.args.device)
                batched_khop_graph = batched_khop_graph.to(self.args.device)

                self.model.train()
                logits_dict= self.model(batched_graph, batched_graph.ndata['feature'], batched_khop_graph, scen='train')
                loss, loss_items = self.get_loss(logits_dict, labels_dict=batched_labels_dict)

                for k in loss_items_total_train:
                    loss_items_total_train[k[0]] += loss_items[k]

                optimizer.zero_grad()
                loss.backward()
                # pcgrad_fn(self.model, losses=loss, optimizer=optimizer)

                optimizer.step()
                # scheduler.step()
                # # The following code is used to record the memory usage
                # py_process = psutil.Process(os.getpid())
                # print(f"CPU Memory Usage: {py_process.memory_info().rss / (1024 ** 3)} GB")
                # print(f"GPU Memory Usage: {torch.cuda.memory_reserved() / (1024 ** 3)} GB")

                # clear GPU cache
                del batched_data
                del batched_graph
                del batched_labels_dict
                del batched_khop_graph
                del logits_dict
                del loss
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                labels_dict_val_mul = {k:[] for k in self.output_route }
                probs_dict_val_mul = {k:[] for k in self.output_route }
                loss_items_total_val = {k:0 for k in self.output_route }
                # eval loop
                for batched_data in self.val_dataloader:
                    batched_graph, batched_labels_dict, batched_khop_graph = batched_data
                    # FIXME: device issue?
                    batched_graph = batched_graph.to(self.args.device)
                    for k,v in batched_labels_dict.items():
                        batched_labels_dict[k] = v.to(self.args.device)
                        if k[0] in self.output_route:
                            labels_dict_val_mul[k[0]].append(v)
                    batched_khop_graph = batched_khop_graph.to(self.args.device)
                    self.model.eval()
                    with torch.no_grad():
                        logits_dict = self.model(batched_graph, batched_graph.ndata['feature'], batched_khop_graph, scen='val')
                        _, loss_items = self.get_loss(logits_dict, labels_dict=batched_labels_dict)
                        for k in self.output_route:
                            loss_items_total_val[k] += loss_items[k]

                        probs = self.get_probs(logits_dict)
                        for k in probs:
                            probs_dict_val_mul[k].append(probs[k])
                    
                    del batched_data
                    del batched_graph
                    del batched_labels_dict
                    del batched_khop_graph
                    del logits_dict
                    del probs
                with torch.no_grad():
                    for k in self.output_route:
                        labels_dict_val_mul[k] = torch.cat([t for t in labels_dict_val_mul[k]])
                        probs_dict_val_mul[k] = torch.cat([t for t in probs_dict_val_mul[k]])
                    # get eval score
                    score_val = self.eval(labels_dict_val_mul, probs_dict_val_mul)
                    del labels_dict_val_mul
                    del probs_dict_val_mul
                    # average different scores
                    score_overall_val = 0
                    for k in self.output_route:
                        score_overall_val += score_val[k][self.args.metric]
                    score_overall_val /= len(self.output_route)
                log_loss(['Train', 'Val'], [loss_items_total_train, loss_items_total_val])
                del loss_items_total_train
                del loss_items_total_val

                # select the best on val set
                if score_overall_val > self.best_score or self.patience_knt > self.args.patience:
                    torch.cuda.empty_cache()
                    self.best_score = score_overall_val
                    self.patience_knt = 0
                    labels_dict_test_mul = {k:[] for k in self.output_route }
                    probs_dict_test_mul = {k:[] for k in self.output_route }
                    loss_items_total_test = {k:0 for k in self.output_route }
                    # eval loop
                    for batched_data in self.test_dataloader:
                        batched_graph, batched_labels_dict, batched_khop_graph = batched_data
                        batched_graph = batched_graph.to(self.args.device)
                        for k,v in batched_labels_dict.items():
                            batched_labels_dict[k] = v.to(self.args.device)
                            if k[0] in self.output_route:
                                labels_dict_test_mul[k[0]].append(v)
                        batched_khop_graph = batched_khop_graph.to(self.args.device)
                        self.model.eval()
                        # get test result
                        with torch.no_grad():
                            logits_dict = self.model(batched_graph, batched_graph.ndata['feature'], batched_khop_graph, scen='test')
                            _, loss_items = self.get_loss(logits_dict, labels_dict=batched_labels_dict)
                            for k in self.output_route:
                                loss_items_total_test[k] += loss_items[k]
                            probs = self.get_probs(logits_dict)
                            for k in probs:
                                probs_dict_test_mul[k].append(probs[k])
                        
                        del batched_data
                        del batched_graph
                        del batched_labels_dict
                        del batched_khop_graph
                        del logits_dict
                        del probs
                    # clear GPU cache
                    for k in self.output_route:
                        labels_dict_test_mul[k] = torch.cat([t for t in labels_dict_test_mul[k]])
                        probs_dict_test_mul[k] = torch.cat([t for t in probs_dict_test_mul[k]])
                    # get test score
                    score_test = self.eval(labels_dict_test_mul, probs_dict_test_mul)
                    del labels_dict_test_mul
                    del probs_dict_test_mul
                    torch.cuda.empty_cache()
                    # log to stdin
                    print(f'Epoch {epoch}: {self.best_score}\n{pprint.pformat(score_test)}')
                    if self.patience_knt > self.args.patience:
                        print("No patience")
                        break
                else:
                    self.patience_knt += 1

        return score_test