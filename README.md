# GNN-Cora-CUHKSZAG
----------------------------------------------------------
## Development Plan:
- Develop general GNN experiment playground in [GNN-for-Node_Classification-AML](https://github.com/Linnore/GNN-for-Node-Classification-AML). 
- [ ] Clearify and Fix the design of inductive learning: For inductive learning, current framework split training graph and validation (testing) graph on edge levels. Though non common edges, some traning nodes are in the validation graph, which may result in higher validation (testing) results.
- [ ] Current mini-batch sampling proposed by GraphSAGE is only enable in negative graph sampling for the linkage prediction task due to the necessity. Should also implement and test mini-batch sampling, even when the full graph can fit into memory (e.g. in the node 



## Future Projects on GNN Applications
GNN-based methods for node classification on various datasets will be implemented in this repo: [GNN-for-Node_Classification-AML](https://github.com/Linnore/GNN-for-Node-Classification-AML). Further focussing on the application of anomaly detection and AML.

----------------------------------------------------------
## Original Commit
This repo contains Pytorch-Lightning implementations of GCN  and GraphSAGE for Node Classification and Link Prediction (as a way for Recommendation System) on the Cora dataset and CUHKSZ-AG dataset.
**Size of this repo:**
~23MB
## Introduction
This project roughly consists of two parts: dataset creation （dataPrep） and GNN application(all jupyter notebooks). **Source codes of our data processing and models are under the folder `./utils`.** 
## To begin with 
1. Install anaconda or any other conda.
2. Create an empty and new environment by `conda create --name yourname python==3.10.9`.
3. Activate the environment you just created.
4. Key libararies required for this project:
Pytorch 2.0, Pytorch Sparse, Pytorch_Lightning 2.0, Pytorch_Geometric, tensorboard
#### Note: if you find any libraries missed when running our code, please check the libraries used at the beginning of the notebook. 
#### About Tensorboard: 
When running the first ipynb notebook which enables the initialization of tensorboard, you need to change the ip address to your own one and choose a port that have not been used in your own computer. When running the remained files, all the results will be shown in the same tensorboard.

Or alternatively, you can launch the tensorbord in the terminal at the root of this repo:
`tensorboard --logdir 'lightning_logs' --port 6003 `
Then visit the tensorboad in the browser: `http://localhost:6003/`.

## Run Our Pipelines
0. The notebook in ./dataPrep is the web crawling procedures.
1. Run everything in `1_NC-Cora_GCN.ipynb` to see the node classification result for Cora dataset using GCN model.
2. Run everything in `2_NC-AG_GCN.ipynb` to see the node classification result for our academic dataset using GCN model.
3. Run everything in `3_NC-Cora_GraphSAGE.ipynb` to see the node classification result for Cora dataset using GraphSAGE model.
4. Run everything in `4_NC-AG_GraphSAGE.ipynb` to see the node classification result for our academic dataset using GraphSAGE model.
5. Run everything in `5_RS-Cora_GCN.ipynb` to see the recommendation result for Cora dataset using GCN model and transductive method.
5. Run everything in `6_RS-AG_GCN.ipynb` to see the recommendation result for our academic dataset using GCN model and transductive method.
6. Run everything in `7_RS-Cora_GraphSAGE_transductive.ipynb` to see the recommendation result for Cora dataset using GraphSAGE model and transductive method.
7. Run everything in `8_RS-AG_GraphSAGE_transductive.ipynb` to see the recommendation result for our academic dataset using GraphSAGE model and transductive method.
8. Run everything in `9_RS-Cora_GraphSAGE_inductive.ipynb` to see the recommendation result for Cora dataset using GraphSAGE model and inductive method.
9. Run everything in `10_RS-AG_GraphSAGE_inductive.ipynb` to see the recommendation result for our academic dataset using GraphSAGE model and inductive method.
## Our formal report on this project
Please refer to the `report.pdf`.
