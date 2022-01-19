import torch
import torch.nn as nn
import inspect

from utilities import filter_kwargs

class LayerFactory:
    """
    Class for singular layer generation from a parameter dictionary
    """
    def __init__(self, layer_fun, use_wrapper=False, **kwargs):
        self.type = kwargs['type'] # do differently
        self.f = layer_fun
        self.kwargs = kwargs
        self.make = self._make
        self.use_wrapper = use_wrapper

        # filter out only the kwargs needed for the layer function
        self.f_kwargs = filter_kwargs(self.kwargs, exclude=['type', 'repeat'])

    class LayerWrapper(nn.Module):
        """
        Inner class wrapper used for debugging
        """
        def __init__(self, layer_fun, *args, **kwargs):
            super(LayerFactory.LayerWrapper, self).__init__()
            self.f = layer_fun(*args, **kwargs)
        def forward(self, *args, **kwargs):
            if torch.is_tensor(args[0].shape):
                print(args[0].shape)
            return self.f(*args, **kwargs)

    class InvalidLayer(Exception):
        pass

    @staticmethod
    def make(conf):
        """
        Static instatiator for a single layer
        """
        LAYER_DICT =   {'linear': nn.Linear, 'conv1d': nn.Conv1d, 'conv2d': nn.Conv2d, 'gru': nn.GRU,
                        'relu': nn.ReLU, 'selu': nn.SELU, 'tanh': nn.Tanh,
                        'batchnorm1d': nn.BatchNorm1d, 'batchnorm2d': nn.BatchNorm2d, 'layernorm': nn.LayerNorm,
                        'dropout': nn.Dropout, 'flatten': nn.Flatten}
        type = conf['type']
        if type in LAYER_DICT:
            f = LAYER_DICT[type]
        else:
            raise LayerFactory.InvalidLayer(type)
        return LayerFactory(f, **conf).make()

    def _make(self):
        """
        Outputs a layer
        "Overrides" static make when called on an instance
        """
        if self.use_wrapper:
            layer = LayerFactory.LayerWrapper(self.f, **self.f_kwargs)
        
            if self.type in ['conv1d', 'conv2d', 'linear']:
                nn.init.kaiming_normal_(layer.f.weight, nonlinearity='linear')
        else:
            layer = self.f(**self.f_kwargs)
        
            if self.type in ['conv1d', 'conv2d', 'linear']:
                nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')

        return layer


class ResidualCell(nn.Module):
    """
    Simple pre-activation residual cell with no batch normalization
    """
    def __init__(self, module_params, rezero=True):
        super(ResidualCell, self).__init__()
        layers = parse_config(module_params)
        self.cell = nn.Sequential(*layers)
        self.alpha = nn.Parameter(torch.Tensor([0] if rezero else [1]), requires_grad=(True if rezero else False))

    def forward(self, x):
        return x + self.alpha * self.cell(x)


def parse_config(module_params):
    """
    Builds a list of pytorch layers from a list of layer parameters (parsed from a configuration file)
    """
    layers = []
    for layer_params in module_params:
        type = layer_params['type']
        repeat = 1 if 'repeat' not in layer_params else layer_params['repeat']
        for i in range(repeat):
            if type == 'sequential':
                layers.append(nn.Sequential(*parse_config(layer_params['cell'])))
            elif type == 'residual':
                layers.append(ResidualCell(layer_params['cell']))
            else:
                layers.append(LayerFactory.make(layer_params))
    return layers
