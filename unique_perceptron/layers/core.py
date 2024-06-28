import numpy as np

class HiddenLayer(object):
    def __init__(self, weights: np.ndarray, biases: np.ndarray, activation: np.ndarray) -> None:
        self.weights = weights
        self.biases = biases
        self.activation = activation()
      
    @staticmethod
    def initialize_weights(input_size: int, neuron_count: int) -> np.ndarray:
        return np.random.randn(input_size, neuron_count) * 0.05
    
    @staticmethod
    def initialize_biases(neuron_count: int) -> np.ndarray:
        return np.zeros((1, neuron_count))       
    
    def forward_propagate(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.z = np.dot(self.input, self.weights) + self.biases
        self.a = self.activation.forward(self.z)
        return self.a

class OutputLayer(object):
    def __init__(self, weights: np.ndarray, biases: np.ndarray, activation: np.ndarray) -> None:
        self.weights = weights
        self.biases = biases
        self.activation = activation()
      
    @staticmethod
    def initialize_weights(input_size: int, neuron_count: int) -> np.ndarray:
        return np.random.randn(input_size, neuron_count) * 0.05
    
    @staticmethod
    def initialize_biases(neuron_count: int) -> np.ndarray:
        return np.zeros((1, neuron_count))       
    
    def forward_propagate(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.z = np.dot(self.input, self.weights) + self.biases
        self.a = self.activation.forward(self.z)
        return self.a
