import numpy as np
from morknn.layers import Layer


class Net:
    def __init__(self, input_dims=None, output_dims=None, activations=None, dtypes=None, names=None):
        if input_dims and output_dims and activations:
            n_layers = len(input_dims)
            names = names if names else np.arange(n_layers)
            dtypes = dtypes if dtypes else [np.float32] * n_layers
            zipper = zip(input_dims, output_dims, activations, dtypes, names)
            self.layers = (
                Layer(
                    input_dim=i_d, output_dim=o_d, activation=act, dtype=dtype, name=name
                ) for i_d, o_d, act, dtype, name in zipper
            )
        else:
            self.layers = None
        
    def define(self, layers):
        self.layers = layers        
        
    def describe(self):
        "same as print"
        print(self)
        
    def _forward_prop(self):
        pass
    
    def _backward_prop(self):
        pass
    
    def train(self, train_x, train_y, learning_rate, epochs, batch_size, verbose=True):
        pass
    
    def validate(self, val_x, val_y, verbose=True):
        pass

    def __repr__(self):
        try:
            name = [str(layer) for layer in self.layers]
            return '\n'.join(name)
        except TypeError as e:
            if self.layers:
                name = [str(layer) for layer in self.layers]
                return str('\n'.join(name))
            else:
                return f'<Net: 0 layers>'
            

if __name__ == '__main__':
    print(Net())
    

    output_dims = [10, 10, 20, 10]
    input_dims = [train_x.shape[0], *output_dims[1:]]
    activations = ['relu', 'relu', 'relu', 'softmax']
    names = ['input', 'hidden_1', 'hidden_2', 'output']
    net = Net(output_dims, input_dims, activations)
    print(net)
    

    net = Net()
    layers = [
        Layer(784, 10, 'relu', dtype=np.float32, name='input'),
        Layer(10, 10, 'relu', dtype=np.float32, name='hidden_1'),
        Layer(20, 20, 'relu', dtype=np.float32, name='hidden_2'),
        Layer(10, 10, 'softmax', dtype=np.float32, name='output'),
    ]
    net.define(layers=layers)
    print(net)