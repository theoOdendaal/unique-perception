
from loss import CategoricalCrossEntropy

class StochasticGradientDescent:
    
    def __init__(self, loss_function, learning_rate: float = 0.01):
        self.loss_function = loss_function
        self.learning_rate = learning_rate