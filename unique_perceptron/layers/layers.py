import numpy as np

class XavierScheme:
    
    @staticmethod
    def uniform_init(shape: tuple[int, int]) -> np.ndarray:
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, shape)
    
    @staticmethod
    def normal_dist_init(shape: tuple[int, int]) -> np.ndarray:
        stddev = np.sqrt(2 / (shape[0] + shape[1]))
        return np.random.normal(0, stddev, shape)

class HiddenLayer(object):
    def __init__(self, weights: np.ndarray, biases: np.ndarray, activation: np.ndarray) -> None:
        self.weights = weights
        self.biases = biases
        self.activation = activation()
        self.x = None # Input
        self.z = None # Pre-activation dot product.
        self.a = None # Post-activation dot product.
      
    @staticmethod
    def initialize_weights(input_size: int, neuron_count: int) -> np.ndarray:
        #return np.random.randn(input_size, neuron_count) * 0.05
        return XavierScheme.uniform_init((input_size, neuron_count)) *0.05
        
        
    @staticmethod
    def initialize_biases(neuron_count: int) -> np.ndarray:
        return np.zeros((1, neuron_count))       
    
    def forward_propagate(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z = np.dot(self.x, self.weights) + self.biases
        self.a = self.activation.forward(self.z)
        return self.a
