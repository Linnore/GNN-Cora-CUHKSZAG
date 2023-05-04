import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.data as geom_data
from torch_geometric.data import Data
import pytorch_lightning as pl
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# define the LightningModule

class GCN(pl.LightningModule):
    """This class define a GCN model for node classification. 
    """
    def __init__(self, dataset, hparams, writer_acc, writer_loss):
        """This class define a GCN model for node classification. 
        Args:
            dataset (torch_geometric.data.Dataset): the dataset that contains one positive graph.
            
            hparams (dict): the dictionary containing hyperparameters.

            writer_acc (lightning.pytorch.loggers.Logger, optional): The summary writer to log training accuracy and validation accuracy on the same plot if provided. Defaults to None.

            writer_loss (lightning.pytorch.loggers.Logger, optional): The summary writer to log training loss and validation loss on the same plot if provided. Defaults to None.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.dataset = dataset
        self.conv1 = GCNConv(hparams["NUM_FEATURES"], 16)
        self.conv2 = GCNConv(16, hparams["NUM_CLASSES"])
        self.dropout_rate = hparams["DROUPOUT_RATE"]
        # writers for tensorboard
        self.writer_acc = writer_acc
        self.writer_loss = writer_loss
        self.train_acc = 0
        self.train_loss = -100

    def forward(self, x, edge_index):
        """The forward path of this model: node features -> GCN layers -> node embedding.
        Note that x and edge_index are from the whole graph.

        Args:
            x (tensor): the tensor with shape (num_nodes, num_features). 

            edge_index (tensor): the tensor with shape (2, num_edges).

        Returns:
            x (tensor): the softmax result of originial node embedding.
        """
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["LEARNING_RATE"])
        return optimizer

    def train_dataloader(self):
        return geom_data.DataLoader(self.dataset, batch_size=1)

    def val_dataloader(self):
        return geom_data.DataLoader(self.dataset, batch_size=1)

    def test_dataloader(self):
        return geom_data.DataLoader(self.dataset, batch_size=1)

    def training_step(self, batch):
        graph = batch[0]
        x, edge_index, y = graph.x, graph.edge_index, graph.y
        out = self.forward(x, edge_index)
        train_loss = F.nll_loss(out[graph.train_mask], y[graph.train_mask].type(torch.LongTensor))
        preds = torch.argmax(out[graph.train_mask], dim=1)
        train_acc = torch.mean((preds == y[graph.train_mask]).float())
        self.train_loss = train_loss
        self.train_acc = train_acc
        self.log("train_acc", train_acc, prog_bar=True)
        return {"loss": train_loss, "acc": train_acc}

    def validation_step(self, batch, batch_idx):
        graph = batch[0]
        x, edge_index, y = graph.x, graph.edge_index, graph.y
        out = self.forward(x, edge_index)
        val_loss = F.nll_loss(out[graph.val_mask], y[graph.val_mask].type(torch.LongTensor))
        self.log("val_loss", val_loss)

        preds = torch.argmax(out[graph.val_mask], dim=1)
        val_acc = torch.mean((preds == y[graph.val_mask]).float())
        self.log("val_acc", val_acc, prog_bar=True)
        self.writer_acc.add_scalars(
            "accuracy",
            {"train_acc": self.train_acc, "val_acc": val_acc},
            self.global_step,
        )
        if self.train_loss != -100:  # step 0
            self.writer_loss.add_scalars(
                "loss",
                {"train_loss": self.train_loss, "val_loss": val_loss},
                self.global_step,
            )
        return {"val_loss": val_loss, "val_accuracy": val_acc}

    def test_step(self, batch, batch_idx):
        graph = batch[0]
        x, edge_index, y = graph.x, graph.edge_index, graph.y
        out = self.forward(x, edge_index)
        loss = F.nll_loss(out[graph.test_mask], y[graph.test_mask].type(torch.LongTensor))
        self.log("test_loss", loss)
        preds = torch.argmax(out[graph.test_mask], dim=1)
        acc = torch.mean((preds == y[graph.test_mask]).float())
        self.log("test_accuracy", acc)
        return {"test_loss": loss, "test_acc": acc}

    def testModel(self, trainer):
        test_result = trainer.test(model=self, verbose=False)
        self.writer_acc.close()
        self.writer_loss.close()
        print("Test accuracy:  %4.2f%%" % (100.0 * test_result[0]["test_accuracy"]))
