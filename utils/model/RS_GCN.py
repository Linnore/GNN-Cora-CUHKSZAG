import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import ReLU
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.data import Data
import pytorch_lightning as pl
import torch_geometric.transforms as T

from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader


class GCN(pl.LightningModule):
    """This class define a GCN model for linkage prediction (recommendation for nodes). 
    """
    def __init__(self, dataset, input_dim, hparams, writer_acc=None, writer_loss=None):
        """This class define a GCN model for linkage prediction (recommendation for nodes). 
        IMPORTANT: After create an object of this class, please do manually call yourobject.data_processing() to prepare the model on your device.

        Args:
            dataset (torch_geometric.data.Dataset): the dataset that contains one positive graph.

            input_dim (int): the length of the node feature vector.

            hparams (dict): the dictionary containing hyperparameters.

            writer_acc (lightning.pytorch.loggers.Logger, optional): The summary writer to log training accuracy and validation accuracy on the same plot if provided. Defaults to None.

            writer_loss (lightning.pytorch.loggers.Logger, optional): The summary writer to log training loss and validation loss on the same plot if provided. Defaults to None.
        """
        super().__init__()
        self.dataset = dataset
        self.pos_data = {}
        self.cpu_pos_data = {}
        self.neg_data = {}
        self.cpu_neg_data = {}

        self.save_hyperparameters(hparams)
        self.batch_size = self.hparams["BATCH_SIZE"]

        self.conv1 = GCNConv(input_dim, 112)
        self.conv2 = GCNConv(112, 16)
        self.dropout_rate = hparams["DROUPOUT_RATE"]

        # writers for tensorboard
        self.writer_acc = writer_acc
        self.writer_loss = writer_loss
        self.train_acc = []
        self.train_loss = []
        self.val_acc = []
        self.val_loss = []

    def forward(self, x, edge_index):
        """The forward path of this model: node features -> GCN layers -> node embedding.
        Note that x and edge_index are from the positive graph.

        Args:
            x (tensor): the tensor with shape (num_nodes, num_features). 

            edge_index (tensor): the tensor with shape (2, num_edges) that defines the edges in COO format.

        Returns:
            embedding (tensor): the tensor with shape (num_nodes, size_embedding).
        """
        embedding = F.dropout(x, p=self.dropout_rate, training=self.training)
        embedding = F.relu(self.conv1(x, edge_index), inplace=True)
        embedding = F.dropout(
            embedding, p=self.dropout_rate, training=self.training)
        embedding = self.conv2(embedding, edge_index)

        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams["LEARNING_RATE"])
        return optimizer

    def train_dataloader(self):
        return NeighborLoader(
            self.cpu_neg_data["train"],
            num_neighbors=self.hparams["NUM_NEIGHBORS"],
            batch_size=self.batch_size,
            input_nodes=self.cpu_neg_data["train"].train_mask,
            transform=T.ToDevice(self.device),
        )

    def training_step(self, batch, batch_idx):
        # x: features of all nodes, edge: all training positive edges
        embedding = self.forward(
            self.pos_data["train"].x, self.pos_data["train"].edge_index)

        # Mapping nodex indices to the whole negative graph for sampled negative edges
        edge_index = torch.zeros(
            batch.edge_index.shape, dtype=torch.int64).to(self.device)
        edge_index[0] = batch.n_id[batch.edge_index[0]]
        edge_index[1] = batch.n_id[batch.edge_index[1]]

        # Stack with all training positive edges
        edge_index = torch.column_stack(
            (self.pos_data["train"].edge_index, edge_index))

        # Collect labels and Compute scores
        edge_label = torch.cat(
            (self.pos_data["train"].edge_label, batch.edge_label))

        scores = F.sigmoid((embedding[edge_index[0]] *
                            embedding[edge_index[1]]).sum(dim=1))

        train_loss = F.binary_cross_entropy(
            scores, edge_label
        )

        train_acc = roc_auc_score(
            edge_label.cpu().detach().numpy(), scores.cpu().detach().numpy())

        tmp_loss = train_loss.detach().clone().item()

        self.train_loss.append(tmp_loss)
        self.train_acc.append(train_acc)

        self.log("train_acc", train_acc, prog_bar=True)
        return{"loss": train_loss}

    def val_dataloader(self):
        return NeighborLoader(
            self.cpu_neg_data["val"],
            num_neighbors=self.hparams["NUM_NEIGHBORS"],
            batch_size=self.batch_size,
            input_nodes=self.cpu_neg_data["val"].val_mask,
            transform=T.ToDevice(self.device),
        )

    def validation_step(self, batch, batch_idx):
        # x: features of all nodes, edge: all positive validation edges
        embedding = self.forward(
            self.pos_data["val"].x, self.pos_data["val"].edge_index)

        # Mapping nodex indices to the whole negative graph for sampled negative edges
        edge_index = torch.zeros(
            batch.edge_index.shape, dtype=torch.int64).to(self.device)
        edge_index[0] = batch.n_id[batch.edge_index[0]]
        edge_index[1] = batch.n_id[batch.edge_index[1]]

        # Stack with all training positive edges
        edge_index = torch.column_stack(
            (self.pos_data["val"].edge_index, edge_index))

        # Collect labels and Compute scores
        edge_label = torch.cat(
            (self.pos_data["val"].edge_label, batch.edge_label))
        scores = F.sigmoid((embedding[edge_index[0]] *
                            embedding[edge_index[1]]).sum(dim=1))

        val_loss = F.binary_cross_entropy(
            scores, edge_label
        )
        val_acc = roc_auc_score(
            edge_label.cpu().detach().numpy(), scores.cpu().detach().numpy())

        self.val_acc.append(val_acc)
        self.val_loss.append(val_loss.item())
        self.log("val_loss", val_loss, on_epoch=True,
                 batch_size=self.batch_size, prog_bar=True)
        self.log("val_acc", val_acc, on_epoch=True,
                 batch_size=self.batch_size, prog_bar=True)

    def on_validation_epoch_end(self) -> None:

        if self.writer_acc != None and len(self.train_loss) > 0:
            train_acc = np.mean(self.train_acc)
            val_acc = np.mean(self.val_acc)
            self.writer_acc.add_scalars(
                "accuracy",
                {"train_acc": train_acc, "val_acc": val_acc},
                self.global_step,
            )

        if self.writer_loss != None and len(self.train_loss) > 0:
            train_loss = np.mean(self.train_loss)
            val_loss = np.mean(self.val_loss)
            self.writer_loss.add_scalars(
                "loss",
                {"train_loss": train_loss, "val_loss": val_loss},
                self.global_step,
            )

        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

        return super().on_validation_epoch_end()

    def test_dataloader(self):
        return NeighborLoader(
            self.cpu_neg_data["test"],
            num_neighbors=self.hparams["NUM_NEIGHBORS"],
            batch_size=self.batch_size,
            input_nodes=self.cpu_neg_data["test"].test_mask,
            transform=T.ToDevice(self.device)
        )

    def test_step(self, batch, batch_idx):
        # x: features of all nodes, edge: all positive testing edges
        embedding = self.forward(
            self.pos_data["test"].x, self.pos_data["test"].edge_index)

        # Mapping nodex indices to the whole negative graph for sampled negative edges
        edge_index = torch.zeros(
            batch.edge_index.shape, dtype=torch.int64).to(self.device)
        edge_index[0] = batch.n_id[batch.edge_index[0]]
        edge_index[1] = batch.n_id[batch.edge_index[1]]

        # Stack with all training positive edges
        edge_index = torch.column_stack(
            (self.pos_data["test"].edge_index, edge_index))

        # Collect labels and Compute scores
        edge_label = torch.cat(
            (self.pos_data["test"].edge_label, batch.edge_label))
        scores = F.sigmoid((embedding[edge_index[0]] *
                            embedding[edge_index[1]]).sum(dim=1))

        test_loss = F.binary_cross_entropy(
            scores, edge_label
        )
        test_acc = roc_auc_score(
            edge_label.cpu().detach().numpy(), scores.cpu().detach().numpy())

        self.val_acc.append(test_acc)
        self.val_loss.append(test_loss.item())
        self.log("test_loss", test_loss, on_epoch=True,
                 batch_size=self.batch_size)
        self.log("test_acc", test_acc, on_epoch=True,
                 batch_size=self.batch_size, prog_bar=True)

        return {"test_loss": test_loss, "test_acc": test_acc}

    def data_processing(self, seed=118010142):
        """ This is the default data processsing method for transductive learning!
        This method prepare the data loaders given the data with proper train-val-test split through properties: train_mask, val_mask, and test_mask.
        Make sure the GCN object received its dataset parameter as a dataset object of torch_geometric.data.Dataset.
        After calling this method, the GCN object would have all the following graphs stored in self.pos_data and self.neg_data:
            the whole positive graph: self.pos_data["all"]
            the training positive subgraph: self.pos_data["train"]
            the validation positive subgraph: self.pos_data["val"]
            the testing positive subgraph: self.pos_data["test"]

            the whole negative graph: self.neg_data["all"]
            the training negative subgraph: self.neg_data["train"]
            the validation negative subgraph: self.neg_data["val"]
            the testing negative subgraph: self.neg_data["test"]
        """

        # The whole positive graph
        data = self.dataset[0]
        data.edge_label = torch.ones(data.num_edges)
        

        # Get split ratio from node masks:
        ratio_train = data.train_mask.sum().cpu().item()/data.num_nodes
        ratio_val = data.val_mask.sum().cpu().item()/data.num_nodes
        ratio_test = data.test_mask.sum().cpu().item()/data.num_nodes
        
        """Process the train-val-test split of the positive graph:
        
        Training graph, Validation graph, and Testing graph are constructed by random edge split
        """
        np.random.seed(seed)
        rand_idx = np.arange(data.num_edges)
        np.random.shuffle(rand_idx)
        data.edge_train_mask = torch.zeros(data.num_edges, dtype=bool)
        data.edge_val_mask = torch.zeros(data.num_edges, dtype=bool)
        data.edge_test_mask = torch.zeros(data.num_edges, dtype=bool)

        data.edge_train_mask[rand_idx[:int(data.num_edges*ratio_train)]] = True
        data.edge_val_mask[rand_idx[int(data.num_edges*ratio_train):int(data.num_edges*(1-ratio_test))]] = True
        data.edge_test_mask[rand_idx[int(data.num_edges*(1-ratio_test)):]] = True
        
        data = data.to(self.device)

        pos_train = Data(
            x=data.x,
            edge_index=data.edge_index[:, data.edge_train_mask],
            edge_label=torch.ones(data.edge_train_mask.sum()),
            train_mask=data.train_mask)
        pos_val = Data(
            x=data.x,
            edge_index=data.edge_index[:, data.edge_val_mask],
            edge_label=torch.ones(data.edge_val_mask.sum()),
            val_mask=data.train_mask)
        pos_test = Data(
            x=data.x,
            edge_index=data.edge_index[:, data.edge_test_mask],
            edge_label=torch.ones(data.edge_test_mask.sum()),
            test_mask=data.train_mask)

        self.pos_data["all"] = data.to(self.device)
        self.pos_data["train"] = pos_train.to(self.device)
        self.pos_data["val"] = pos_val.to(self.device)
        self.pos_data["test"] = pos_test.to(self.device)

        # Construct the negative graph
        neg_data = data.clone()

        u, v = data.edge_index
        value = torch.ones(len(u)).to(self.device)
        adjmatrix = torch.sparse_coo_tensor(data.edge_index, value)
        adj_neg = 1 - adjmatrix.to_dense() - torch.eye(data.num_nodes).to(self.device)
        neg_u, neg_v = torch.where(adj_neg != 0)
        neg_edge_index = torch.stack((neg_u, neg_v))

        neg_data.edge_index = neg_edge_index
        neg_data.edge_label = torch.zeros(neg_data.num_edges)

        """Process the train-val-test split of the positive graph:
        
        Training graph, Validation graph, and Testing graph are constructed by random edge split
        """
        np.random.seed(seed)
        rand_idx = np.arange(neg_data.num_edges)

        np.random.shuffle(rand_idx)
        neg_data.edge_train_mask = torch.zeros(neg_data.num_edges, dtype=bool)
        neg_data.edge_val_mask = torch.zeros(neg_data.num_edges, dtype=bool)
        neg_data.edge_test_mask = torch.zeros(neg_data.num_edges, dtype=bool)

        neg_data.edge_train_mask[rand_idx[:int(neg_data.num_edges*ratio_train)]] = True
        neg_data.edge_val_mask[rand_idx[int(neg_data.num_edges*ratio_train):int(neg_data.num_edges*(1-ratio_test))]] = True
        neg_data.edge_test_mask[rand_idx[int(neg_data.num_edges*(1-ratio_test)):]] = True
        

        neg_train = Data(
            x=neg_data.x,
            edge_index=neg_data.edge_index[:, neg_data.edge_train_mask],
            edge_label=torch.zeros(neg_data.edge_train_mask.sum()),
            train_mask=neg_data.train_mask)
        
        neg_val = Data(
            x=neg_data.x,
            edge_index=neg_data.edge_index[:, neg_data.edge_val_mask],
            edge_label=torch.zeros(neg_data.edge_val_mask.sum()),
            val_mask=neg_data.train_mask)

        neg_test = Data(
            x=neg_data.x,
            edge_index=neg_data.edge_index[:, neg_data.edge_test_mask],
            edge_label=torch.zeros(neg_data.edge_test_mask.sum()),
            test_mask=neg_data.train_mask)

        self.neg_data["all"] = neg_data.to(self.device)
        self.neg_data["train"] = neg_train.to(self.device)
        self.neg_data["val"] = neg_val.to(self.device)
        self.neg_data["test"] = neg_test.to(self.device)

        # Store the cpu version of the datasets:
        self.cpu_pos_data["all"] = data.clone().cpu()
        self.cpu_pos_data["train"] = pos_train.clone().cpu()
        self.cpu_pos_data["val"] = pos_val.clone().cpu()
        self.cpu_pos_data["test"] = pos_test.clone().cpu()

        self.cpu_neg_data["all"] = neg_data.clone().cpu()
        self.cpu_neg_data["train"] = neg_train.clone().cpu()
        self.cpu_neg_data["val"] = neg_val.clone().cpu()
        self.cpu_neg_data["test"] = neg_test.clone().cpu()

        print("Data Processing Done on:", self.device)

