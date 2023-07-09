import numpy as np
from morknn.layers import Layer


class Net:
    def __init__(self, input_dims=None, output_dims=None, activations=None, dtypes=None, names=None):
        if input_dims and output_dims and activations:
            n_layers = len(input_dims)
            names = names if names else np.arange(n_layers)
            dtypes = dtypes if dtypes else [np.float32] * n_layers
            zipper = zip(input_dims, output_dims, activations, dtypes, names)
            self.layers = [
                Layer(
                    input_dim=i_d, output_dim=o_d, activation=act, dtype=dtype, name=name
                ) for i_d, o_d, act, dtype, name in zipper
            ]
        else:
            self.layers = None
        
    def define(self, layers):
        self.layers = layers
        
    def describe(self):
        "same as print"
        print(self)
        
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
        
    def one_hot(self, data_y):
        n_labels = data_y.shape[0]
        n_options = int(data_y.max())
        one_hoty = np.zeros((n_labels, n_options + 1))
        one_hoty[np.arange(n_labels), data_y] = 1
        return one_hoty.transpose()

    def _forward_prop(self, data_x):
        assert self.layers is not None
        for layer in self.layers:
            print(layer.weights.shape, data_x.shape)
            data_z = matmul(layer.weights, data_x) + layer.bias
            data_x = layer.activation._act()(data_z)
        return data_z, data_x

    def _backward_prop(self, data_x, data_y, data_z):
        assert self.layers is not None
        for n, layer in enumerate(reversed(self.layers)):
            
            if n == 0:
                d_z = data_z - data_y
                print(f'{n}: d_z={d_z.shape}, data_x.T={data_x.transpose().shape}')
                delta_weights = self.normalize_factor * matmul(d_z, data_x.transpose())
            else:
                d_z = matmul(self.layers[::-1][n-1].weights.transpose(), d_z) * layer.activation._d_act()(data_z)
                
                print(f'{n}: d_z={d_z.shape}, data_x.T={data_x.transpose().shape}')
                delta_weights = self.normalize_factor * matmul(d_z, data_x.transpose())
            
            delta_bias = self.normalize_factor * np.sum(delta_weights)
            
            print(f'{n}: weights={layer.weights.shape}, d_weights={delta_weights.shape}')
            layer.weights = layer.weights - self.learning_rate * delta_weights
            layer.bias = layer.bias - self.learning_rate * delta_bias
    
    def train(self, data_x, data_y, learning_rate, epochs, verbose=True):
        self.normalize_factor = 1 / data_x.shape[0]
        self.learning_rate = learning_rate
        data_y = self.one_hot(data_y)
        for epoch in range(epochs):
            data_z, data_x = self._forward_prop(data_x)
            self._backward_prop(data_x, data_y, data_z)
            if epoch % 10 == 0:
                pred = self.get_predictions(data_x)
                accuracy = self.get_accuracy(pred, data_y)
                print(f"EPOCH: {epoch:03d}, accuracy: {accuracy:.4f}")
    
    def validate(self, val_x, val_y, verbose=True):
        pass
    
    def get_predictions(self, A):
        pred = np.argmax(A, 0)
        return pred

    def get_accuracy(self, predictions, data_y):
        accuracy = np.sum(predictions == data_y) / data_y.shape[0]
        return accuracy


# class Net:
#     def __init__(self, input_dims=None, output_dims=None, activations=None, dtypes=None, names=None):
#         if input_dims and output_dims and activations:
#             n_layers = len(input_dims)
#             names = names if names else np.arange(n_layers)
#             dtypes = dtypes if dtypes else [np.float32] * n_layers
#             zipper = zip(input_dims, output_dims, activations, dtypes, names)
#             self.layers = (
#                 Layer(
#                     input_dim=i_d, output_dim=o_d, activation=act, dtype=dtype, name=name
#                 ) for i_d, o_d, act, dtype, name in zipper
#             )
#         else:
#             self.layers = None
        
#     def define(self, layers):
#         self.layers = layers        
        
#     def describe(self):
#         "same as print"
#         print(self)
        
#     def _forward_prop(self):
#         pass
    
#     def _backward_prop(self):
#         pass
    
#     def train(self, train_x, train_y, learning_rate, epochs, batch_size, verbose=True):
#         pass
    
#     def validate(self, val_x, val_y, verbose=True):
#         pass

#     def __repr__(self):
#         try:
#             name = [str(layer) for layer in self.layers]
#             return '\n'.join(name)
#         except TypeError as e:
#             if self.layers:
#                 name = [str(layer) for layer in self.layers]
#                 return str('\n'.join(name))
#             else:
#                 return f'<Net: 0 layers>'
            

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