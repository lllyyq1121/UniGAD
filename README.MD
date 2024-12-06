# UniGAD: Unifying Multi-level Graph Anomaly Detection

This is the official implementation of the following paper:

>UniGAD: Unifying Multi-level Graph Anomaly Detection  
>Yiqing Lin, Jianheng Tang, Chenyi Zi, H.Vicky Zhao, Yuan Yao, Jia Li  
>NeurIPS 2024 

Environment Setup
-----------------

Before you start, install Pytorch and DGL with appropriate CUDA support. Please refer to the PyTorch and DGL websites for the specific guidelines.

Install dependencies:

```shell
pip install -r requirements.txt
```

Datasets
-----------------
We release a real-world large-scale social group dataset T-Group in the paper, which contains both node anomaly labels and graph anomaly labels. The dataset is on [google drive](https://drive.google.com/file/d/1B-pmATZt9aBmxD8PkuUFAP6ZI-939Xma/view?usp=sharing). Download and unzip the file in the dataset folder.



Experiments
-----------------

Parameters:
- datasets: dataset ID
- batch_size: batch size of graphs
- cross_mode: cross traning mode for mult-level tasks, default = "ng2ng"
- khop: MRQSampler tree depth, default = 1
- kernels: the model used in the pretrained encoder, default = gcn
- epoch_pretrain: number of pretrain epochs, default = 50
- lr: learning rate of the pretraning, default = 5e-4
- epoch_ft: number of GraphStitch Network traning epochs, default = 200
- lr_ft: learning rate of the GraphStitch Network, default = 0.003
- trials: number of experiment repetitions, default = 1

Example: 
```shell
mkdir results/
cd src/
# reproduce the results on dataset weibo
python main.py --datasets 1 --pretrain_model graphmae --kernels bwgnn,gcn --lr 5e-4 --save_model --epoch_pretrain 50 --batch_size 1 --khop 1 --epoch_ft 300 --lr_ft 0.003 --final_mlp_layers 3 --cross_modes ne2ne,n2ne,e2ne --metric AUROC --trials 5
```

>Note: In our experiments, we found that typically hop = 1 or 2 is optimal, so the MRQSampler in our code is implemented as two unfolded versions dedicated to orders 1 and 2 rather than a recursive version for the performance.


## Reference

### Encoder Model

[GraphMAE](https://github.com/THUDM/GraphMAE) (https://github.com/THUDM/GraphMAE).

### Pareto loss function

[Multi-task loss optimizer](https://github.com/tomtang110/multi-task_loss_optimizer/tree/master) (https://github.com/tomtang110/multi-task_loss_optimizer/tree/master)

### Baseline Information

[Node and Edge level Baselines](https://github.com/squareRoot3/GADBench/) (https://github.com/squareRoot3/GADBench/).

[OCGIN,OCGTL](https://github.com/boschresearch/GraphLevel-AnomalyDetection) (https://github.com/boschresearch/GraphLevel-AnomalyDetection).

[GLocalKD](https://github.com/RongrongMa/GLocalKD) (https://github.com/RongrongMa/GLocalKD).

[iGAD](https://github.com/graph-level-anomalies/iGAD/tree/main) (https://github.com/graph-level-anomalies/iGAD/tree/main).

[GmapAD](https://github.com/XiaoxiaoMa-MQ/GmapAD) (https://github.com/XiaoxiaoMa-MQ/GmapAD).

[RQGNN](https://github.com/xydong127/RQGNN/tree/main) (https://github.com/xydong127/RQGNN/tree/main).

[GraphPrompt](https://github.com/Starlien95/GraphPrompt) (https://github.com/Starlien95/GraphPrompt)

[All in One](https://github.com/sheldonresearch/ProG) (https://github.com/sheldonresearch/ProG)
