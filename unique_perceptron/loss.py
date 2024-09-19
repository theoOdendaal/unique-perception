from abc import ABC, abstractmethod

import numpy as np

class LossFunction(ABC):
    
    @abstractmethod
    def cost(self, y, pred):
        pass

    @abstractmethod
    def cost_derivative(self, y, pred):
        pass

"""
class CategoricalCrossEntropy:
    def cost(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        self.cost = -np.log(np.max(np.multiply(y,pred)))
        return self.cost # DERIVATIVE OF THE CCE function is simply 1. Therefore, the loss will simply be multiplied by 1
    
class MeanSquaredError:
    def cost(y: np.ndarray, pred: np.ndarray):
        return np.mean((y - pred) ** 2)
    
    def cost_derivative(y: np.ndarray, pred: np.ndarray):
        return 2 * (pred - y) / y.size
"""

class MeanSquaredError(LossFunction):
    
    def cost(self, targets, predictions):
        num_samples = predictions.shape[0]
        self.loss = np.sum((predictions - targets) ** 2) / num_samples
        return self.loss
  
    def cost_derivative(self, target, predicted_values):
        num_samples = target.shape[0]
        self.gradient = 2 * (predicted_values - target) / num_samples
        return self.gradient
    

class CategoricalCrossEntropy(LossFunction):
    
    def cost(self, targets, predictions):
        num_samples = predictions.shape[0]
        epsilon = 1e-7
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        self.loss = -np.sum(targets * np.log(predictions + 1e-9)) / num_samples
        return self.loss
    
    def cost_derivative(self, actual_labels, predicted_probs):
        num_samples = actual_labels.shape[0]
        epsilon = 1e-7
        self.gradient = -(actual_labels / (predicted_probs + epsilon)) / num_samples
        return self.gradient