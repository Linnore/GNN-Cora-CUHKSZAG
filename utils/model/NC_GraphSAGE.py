import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import ReLU
from torch_geometric.nn import SAGEConv, aggr, Sequential
import pytorch_lightning as pl
import torch_geometric.data as geom_data
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter


class GraphSAGE(pl.LightningModule):
    """This class define a GCN model for node classification. 
    """
    def __init__(self, dataset, input_dim, hparams, writer_acc=None, writer_loss=None):
        """This class define a GCN model for node classification. 
        
        Args:
            dataset (torch_geometric.data.Dataset): the dataset that contains one positive graph.

            input_dim (int): the length of the node feature vector.

            hparams (dict): the dictionary containing hyperparameters.

            writer_acc (lightning.pytorch.loggers.Logger, optional): The summary writer to log training accuracy and validation accuracy on the same plot if provided. Defaults to None.

            writer_loss (lightning.pytorch.loggers.Logger, optional): The summary writer to log training loss and validation loss on the same plot if provided. Defaults to None.
        """
        super().__init__()
        self.dataset = dataset
        self.save_hyperparameters(hparams) 
        self.aggregator_type = hparams["AGGREGATOR_TYPE"]

        if self.aggregator_type == 'mean':
            self.aggr = aggr.MeanAggregation()
        if self.aggregator_type == 'max':
            self.aggr = aggr.MaxAggregation()
        if self.aggregator_type == 'softmax':
            self.aggr = aggr.SoftmaxAggregation(learn=True)
        if self.aggregator_type == 'lstm':
            in_channels=np.arange(10,20,1)
            out_channels=np.arange(10,20,1)
            self.aggr = aggr.LSTMAggregation(in_channels=in_channels[0], out_channels=out_channels[0])
        if self.aggregator_type == 'power':  
            self.aggr = aggr.PowerMeanAggregation(learn=True)
        if self.aggregator_type == 'sort':
            self.aggr = aggr.SortAggregation(k=7) 
            
        self.model = Sequential("x, edge_index", [
            (SAGEConv(input_dim, hparams["HIDDEN_DIM"]
             [0], aggr=self.aggr), "x, edge_index -> x"),
            (ReLU(inplace=True)),
            (SAGEConv(hparams["HIDDEN_DIM"][0], hparams["HIDDEN_DIM"]
                      [1], aggr=self.aggr), "x, edge_index -> x")
        ])
        
        self.writer_acc = writer_acc
        self.writer_loss = writer_loss
        self.train_acc = 0
        self.train_loss = -100

    def forward(self, x, edge_index):
        x = self.model(x, edge_index)
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
        self.train_loss = F.nll_loss(out[graph.train_mask], y[graph.train_mask].type(torch.LongTensor))
        preds = torch.argmax(out[graph.train_mask], dim=1)
        self.train_acc = torch.mean((preds == y[graph.train_mask]).float())
        return {'loss': self.train_loss}
    
    def validation_step(self, batch, batch_idx):
        graph = batch[0]
        x, edge_index, y = graph.x, graph.edge_index, graph.y
        out = self.forward(x, edge_index)
        val_loss = F.nll_loss(out[graph.val_mask], y[graph.val_mask].type(torch.LongTensor))
        self.log('val_loss', val_loss)
        preds = torch.argmax(out[graph.val_mask], dim=1)
        val_acc = torch.mean((preds == y[graph.val_mask]).float())
        self.log('validation_accuracy', val_acc)
        self.writer_acc.add_scalars('accuracy', {"train_acc":self.train_acc, 
                                                   "val_acc":val_acc},
                                    self.global_step)
        if self.train_loss != -100:
            self.writer_loss.add_scalars('loss', {"train_loss": self.train_loss,
                                              "val_loss": val_loss}, self.global_step)
        return {'val_loss': val_loss, 'val_acc': val_acc}

    
    def test_step(self, batch, batch_idx):
        graph = batch[0]
        x, edge_index, y = graph.x, graph.edge_index, graph.y
        out = self.forward(x, edge_index)
        loss = F.nll_loss(out[graph.test_mask], y[graph.test_mask].type(torch.LongTensor))
        self.log('test_loss', loss)
        preds = torch.argmax(out[graph.test_mask], dim=1)
        acc = torch.mean((preds == y[graph.test_mask]).float())
        self.log('test_accuracy', acc)
        return {'test_loss': loss}
    
    def testModel(self,trainer):
        test_result = trainer.test(self,verbose=False)
        self.writer_acc.close()
        self.writer_loss.close()
        print("Test accuracy:  %4.2f%%" % (100.0 * test_result[0]["test_accuracy"]))

