import numpy as np

class CategoricalCrossEntropy(object):
    def get(self, y: np.ndarray, pred: np.ndarray) -> np.ndarray:
        self.cost = -np.log(np.max(np.multiply(y,pred)))
        return self.cost ###################DERIVATIVE OF THE CCE function is simply 1!!!!!. Therefore, the loss will simply be multiplied by 1