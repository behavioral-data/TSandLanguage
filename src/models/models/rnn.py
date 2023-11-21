import torch.nn as nn

from src.models.models.bases import ClassificationModel

""" 
Borrowed from tsai
"""

class _RNN_Base(nn.Module):
    def __init__(self, c_in, hidden_size=100, n_layers=1, bias:bool=True, bidirectional:bool=False):
        super().__init__()
        self.rnn = self._cell(c_in, hidden_size, num_layers=n_layers, bias=bias, batch_first=True,
                              bidirectional=bidirectional)
                

    def forward(self, x): 
        output, _ = self.rnn(x) # output from all sequence steps: [batch_size x seq_len x hidden_size * (1 + bidirectional)]
        output = output[:, -1]  # output from last sequence step : [batch_size x hidden_size * (1 + bidirectional)]
        return output
    
    def _weights_init(self, m): 
        # same initialization as keras. Adapted from the initialization developed 
        # by JUN KODA (https://www.kaggle.com/junkoda) in this notebook
        # https://www.kaggle.com/junkoda/pytorch-lstm-with-tensorflow-like-initialization
        for name, params in m.named_parameters():
            if "weight_ih" in name: 
                nn.init.xavier_normal_(params)
            elif 'weight_hh' in name: 
                nn.init.orthogonal_(params)
            elif 'bias_ih' in name:
                params.data.fill_(0)
                # Set forget-gate bias to 1
                n = params.size(0)
                params.data[(n // 4):(n // 2)].fill_(1)
            elif 'bias_hh' in name:
                params.data.fill_(0)
        
class RNN(_RNN_Base):
    _cell = nn.RNN
    
class LSTM(_RNN_Base):
    _cell = nn.LSTM
    
class GRU(_RNN_Base):
    _cell = nn.GRU


class RNNForClassification(ClassificationModel):
    def __init__(self,cell_type:str="RNN", hidden_size:int = 100,
                 num_classes:int=2, bidirectional:bool = False,
                 n_layers:int = 1,
                **kwargs):
        
        super().__init__(**kwargs)
        _n_timesteps, n_features = kwargs.get("input_shape")
        cell_args = dict(c_in=n_features,
                         hidden_size=hidden_size, 
                         n_layers=n_layers,
                         bidirectional=bidirectional)
        
        
        if cell_type == "RNN":
            self.encoder = RNN(**cell_args)
        elif cell_type == "LSTM":
            self.encoder = LSTM(**cell_args)
        elif cell_type == "GRU":
            self.encoder = GRU(**cell_args)
        else:
            raise ValueError("cell_type must be one of RNN, LSTM or GRU")
            
        
        self.name= cell_type if not bidirectional else "Bi-" + cell_type

        self.fc = nn.Linear(hidden_size * (1 + bidirectional), num_classes)
        self.criteron = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        
    def forward(self, inputs_embeds, label, **kwargs):
        x = self.encoder(inputs_embeds)
        x = self.fc(x)
        loss = self.criteron(x, label)
        return loss, x
