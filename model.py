import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from modules import LayerFactory, parse_config

from config import CONFIG


class ConvolutionalEncoder(nn.Module):
    """
    CNN stack to convert 2D input to a 1-dimensional embedding
    """
    def __init__(self):
        super(ConvolutionalEncoder, self).__init__()

        module_params = CONFIG['model']['2d_convolutional_encoder']
        
        self.encoder = nn.Sequential(*parse_config(module_params))

    def forward(self, x):
        return self.encoder(x)


class ConvolutionalEncoder1D(nn.Module):
    """
    CNN stack to convert 2D input to a 1-dimensional embedding by convolutions along the x dimension
    """
    def __init__(self):
        super(ConvolutionalEncoder1D, self).__init__()

        module_params = CONFIG['model']['1d_convolutional_encoder']
        
        self.stack = nn.Sequential(*parse_config(module_params['stack']))
        self.embedding = nn.Sequential(*parse_config(module_params['embedding']))

    def forward(self, x):
        x = self.stack(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.embedding(x)
        return x


class CRNN_Classifier(nn.Module):
    """
    Convolutional Recurrent NN to assign a label to a sequence of 2D inputs
    """
    def __init__(self):
        super(CRNN_Classifier, self).__init__()
        
        self.rnn_params = CONFIG['model']['crnn']['rnn']
        clsf_params = CONFIG['model']['crnn']['classifier']
        self.cnn = ConvolutionalEncoder1D()
        self.rnn = LayerFactory.make(self.rnn_params)
        self.classifier = nn.Sequential(*parse_config(clsf_params))
        self.init_hidden = nn.Parameter(torch.randn(self.rnn_params['num_layers'], self.rnn_params['hidden_size']), requires_grad=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, x_lens=None, h0=None):
        batch_size = x.shape[0]
        max_len = x.shape[1]

        # if no initial hidden state is supplied, use the initial state repeated along the batch axis
        if h0 is None: h0 = self.init_hidden.unsqueeze(1).repeat(1, batch_size, 1)

        # pack sequences to optimize the handling of varing lengths
        packed_x = pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        # perform encoding on the non-padding data
        x_enc = self.cnn(packed_x.data)

        # repack the sequence to be passed to the RNN
        packed_x = PackedSequence(x_enc, packed_x.batch_sizes, packed_x.sorted_indices, packed_x.unsorted_indices)

        # pass packed data through the RNN
        packed_x, h_n = self.rnn(packed_x, h0)

        # unpack the results
        x, x_lens = pad_packed_sequence(packed_x, batch_first=True, total_length=max_len)

        # get the last output of each sequence in the batch
        x_n = x[torch.arange(batch_size), x_lens - 1]

        # pass the rnn output through the classification layer
        y_pred = self.classifier(x_n)

        return y_pred, h_n