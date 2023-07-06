import numpy as np
from morknn.layers import Layer


class Net:
    def __init__(self, input_dims, output_dims, activations, dtypes=None, names=None):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.activations = activations
        self.n_layers = len(input_dims)
        self.names = names if names else np.arange(self.n_layers)
        self.dtypes = dtypes if dtypes else [np.float32] * self.n_layers
        zipper = zip(
            self.input_dims, self.output_dims, self.activations, self.dtypes, self.names
        )
        self.layers = (
            Layer(
                input_dim=i_d, output_dim=o_d, activation=act, dtype=dtype, name=name
            ) for i_d, o_d, act, dtype, name in zipper
        )
        
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
        name = [str(next(self.layers)) for _ in range(self.n_layers)]
        return str('\n'.join(name))

    
if __name__ == '__main__':
    output_dims = [10, 10, 20, 10]
    input_dims = [train_x.shape[0], *output_dims[1:]]
    activations = ['relu', 'relu', 'relu', 'softmax']
    names = ['input', 'hidden_1', 'hidden_2', 'output']

    net = Net(input_dims, output_dims, activations, names=names)
    net.describe()