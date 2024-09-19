import numpy as np

class ReLu(object):       
    def forward(self, s: np.ndarray) -> np.ndarray:
        self.a = np.maximum(0, s)
        return self.a
        
    def backward(self,s: np.ndarray) -> np.ndarray:
        self.d_a = np.where(s > 0, 1, 0)
        return self.d_a

class LeakyReLu(object):      
    def __init__(self, slope: float = 0.01) -> None:
        self.slope = slope
            
    def forward(self,s: np.ndarray) -> np.ndarray:
        self.a = np.where(s > 0, s, s * self.slope)
        return self.a
        
    def backward(self,s: np.ndarray) -> np.ndarray:
        self.d_a = np.where(s > 0, 1, self.slope)
        return self.d_a
        
class Sigmoid(object):
    def forward(self, s: np.ndarray) -> np.ndarray:
        self.a = 1 / (1 + np.exp(-s))
        return self.a

    def backward(self, s: np.ndarray) -> np.ndarray:
        self.d_a = s * (1 - s)
        return self.d_a

class SoftMax(object):
    def forward(self, s: np.ndarray) -> np.ndarray:
        self.a = np.exp(s - np.max(s, axis=1, keepdims=True))
        self.a = self.a / np.sum(self.a, axis=1, keepdims=True)
        return self.a

    def backward(self, s: np.ndarray) -> np.ndarray:
        #self.d_a = np.diagflat(s) - np.dot(s, s.T) 
        #return self.d_a 
        return s