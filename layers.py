import numpy as np


class Layer():
    def __init__(self, input_dim: int, output_dim: int, activation: str, dtype=np.float32, name='', **kw):
        """
            Creates a layer with weights, bias, activation, name
            
             input_dim (int): the number of nodes in the layer (analogous to "rows")
            output_dim (int): the number of nodes in the next layer (analogous to columns)
            activation (str): the activation function used on the layer
                dtype (type): the dtype for the weights and bias (default np.float32)
                  name (str): (optional) give the layer a name (does not do anything)
                        **kw: pass optional keyword arguments (like "alpha" for leaky_relu)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activator_name = activation
        self.dtype = dtype
        self.name = name
        
        self.lo = -1
        self.hi = 1
        self.weights = self._weights()
        self.bias = self._bias()
        self.activation = self._activator(**kw)
        
    def _weights(self):
        shape = (self.input_dim, self.output_dim)
        return np.random.uniform(self.lo, self.hi, shape).astype(self.dtype)

    def _bias(self):
        shape = (self.input_dim, 1)
        return np.random.uniform(self.lo, self.hi, shape).astype(self.dtype)
    
    def _activator(self, **kw):
        return Activator(self.activator_name, **kw)
    
    def __repr__(self):
        name = f"""LAYER:
                   name: '{self.name}'
            weights_dim: {self.weights.shape}
               bias_dim: {self.bias.shape}
                  dtype: {self.dtype}
             activation: {self.activation}
        """.replace('            ', ' - ')
        return name
    

if __name__ == '__main__':
    l = Layer(20, 10, 'relu', name='input')
    print(l)