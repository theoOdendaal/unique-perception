
class Accuracy(object):
    def __init__(self) -> None:
        self.correct = 0
        self.count = 0
    
    def increase_correct(self) -> None:
        self.correct += 1
    
    def increase_count(self) -> None:
        self.count += 1
        
    def evaluate(self, y, pred) -> None:
    #y : Actual output.
    #pred : Predicted output.
        self.count += len(y)
        for k1, k2 in zip(y,pred):
            if k1 == k2:
                self.increase_correct()
    
    def set_zero(self) -> None:
        self.correct = 0
        self.count = 0
        
    def get(self) -> float:
        return self.correct / self.count