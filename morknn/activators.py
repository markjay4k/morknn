import numpy as np


class Activator:
    def __init__(self, name, **kw):
        self.name = name.lower()
        self.activation = self._act()
        self.d_activation = self._d_act()
        self.alpha = kw['alpha'] if 'alpha' in kw.keys() else 0.01
    
    def relu(self, z):
        return np.maximum(0, z)

    def d_relu(self, z):
        return z >= 0
    
    def leaky_relu(self, z):
        return np.maximum(self.alpha * z, z)

    def d_leaky_relu(self, z):
        a = z.copy()
        a[a < 0] = self.alpha
        a[a > self.alpha] = 1
        return a
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def d_sigmoid(self, z):
        return sigmoid(z) * (1 - sigmoid(z))

    def softmax(self, z):
        return np.exp(z) / sum(np.exp(z))
    
    def _act(self):
        
        if self.name == 'relu':
            return self.relu
        if self.name == 'leaky_relu':
            return self.leaky_relu
        if self.name == 'sigmoid':
            return self.sigmoid
        if self.name == 'softmax':
            return self.softmax
        else:
            raise ValueError(f'activation {self.name} not implemented')
        
    def _d_act(self):
        """
            return the derivative activation function
        """
        if self.activation == self.relu:
            return self.d_relu
        if self.activation == self.leaky_relu:
            return self.d_leaky_relu
        if self.activation == self.sigmoid:
            return self.d_sigmoid
        if self.activation == self.softmax:
            # TODO: add derivative of softmax so it can be used like other layers
            #       Need to figure out how to deal with the derivative when it's
            #       the output layer
            return None
        else:
            raise ValueError(f'activation {self.name} not implemented')
            
    def __repr__(self):
        name = f"<Activation: '{self.name}'>"
        return name
    

if __name__ == '__main__':
    activation_options = ['relu', 'leaky_relu', 'sigmoid', 'softmax']
    for ao in activation_options:
        a = Activator(ao)
        print(a)
        print(f'  {a._act()}')    
        print(f'  {a._d_act()}')