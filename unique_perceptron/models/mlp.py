import numpy as np
import pickle

from layers.core import HiddenLayer
from layers.activation import ReLu, LeakyReLu, Sigmoid, SoftMax
from metrics import Accuracy
from loss import CategoricalCrossEntropy
from optimizers.sgd import StochasticGradientDescent

class MultiLayerPerceptron(object): 
    def __init__(
        self,
        lr=0.01,
        metric=Accuracy,
        loss_function=CategoricalCrossEntropy
        ) -> None:
        
        self.metric = metric()
        self.loss_function = loss_function()
        self.hidden_layers = []
        self.learning_rate = lr
    
    def add_dense(self, weights, biases, activation) -> None:      
        self.hidden_layers.append(HiddenLayer(weights,biases,activation))
                    
    def evaluate(self, input, output) -> None:
        self.set_data(input, output)
        self.metric.set_zero()
        print(f'Actual output | {self.get_prediction(output)}')
        print(f'Predicted output | {self.get_prediction(self.forward_propagate())}')
        print(f'{self.get_statistics()}')
        self.metric.set_zero()
    
    def set_data(self, input, output) -> None:
        self.input = input
        self.actual_output = output 
       
    def forward_propagate(self) -> np.ndarray:
        self.predicted_output = self.input
        for z in self.hidden_layers:
            self.predicted_output = z.forward_propagate(self.predicted_output)
        return self.predicted_output
     
    def backward_propogate(self) -> None:      
        self.hidden_layers.reverse()
        self.d_weights = []
        self.d_biases = []
        self.dz = self.predicted_output - self.actual_output
        
        for i in range(len(self.hidden_layers)):
            n = self.hidden_layers[i]
            if i != 0:
                self.w = self.hidden_layers[i-1].weights
                self.dz = np.dot(self.dz, self.w.T) * n.activation.backward(n.a)
            self.dw = np.dot(self.dz.T, n.input)
            self.db = np.sum(self.dz, axis=0, keepdims=True)
        self.d_weights.append(self.dw)
        self.d_biases.append(self.db)
        
        self.hidden_layers.reverse()
        self.d_weights.reverse()
        self.d_biases.reverse()
        
        for w,b,n in zip(self.d_weights, self.d_biases, self.hidden_layers):
            n.weights = n.weights - self.learning_rate * w.T
            n.biases = n.biases - self.learning_rate * b
        

       
    def fit_model(self) -> None:
        self.forward_propagate()
        self.backward_propogate()
        
    def optimize_model(self, input, output, iterations, update_interval, cycles,batch=True) -> None:
        self.metric.set_zero()
        if batch:
            for c in range(cycles):
                for batch, (i, o) in enumerate(zip(input,output)):
                    self.set_data(i, o)
                    for epoch in range(1,iterations+1):
                        self.fit_model()
                        if max(epoch,1) % (iterations/update_interval) == 0:
                            print(f'Cycle: {c+1}/{cycles}\t | Batch: {batch+1}/{len(input)}\t | Epoch: {epoch}\t | {self.get_statistics()}')
        else:
            for c in range(cycles):
                self.set_data(input,output)
                for epoch in range(1, iterations+1):
                    self.fit_model()
                    if max(epoch,1) % (iterations/update_interval) == 0:
                        print(f'Cycle: {c+1}/{cycles}\t | Epoch: {epoch}\t | {self.get_statistics()}')
        self.metric.set_zero()
  
    def get_prediction(self,x) -> np.ndarray:
        if len(x.shape) != 1:
            return np.argmax(x,axis=1)
        else:
            return np.array(np.argmax(x)).reshape(1,1)

    def get_confidence(self) -> float:
        # Returns the confidence of the predicted output.
        return np.max(np.multiply(self.actual_output,self.predicted_output),axis=1)
    
    def get_statistics(self) -> str:
        # Returns the current network loss, prediction confidence, and specified metric.
        # Requires at least one iteration of the forward_propagate function.
        self.loss_function.get(self.actual_output,self.predicted_output)
        self.metric.evaluate(self.get_prediction(self.actual_output), self.get_prediction(self.predicted_output)) 
        return (f'Loss: {np.round(self.loss_function.cost,6)}\t | Confidence: {np.round(np.mean(self.get_confidence()),6)}\t | {type(self.metric).__name__}: {np.round(self.metric.get(),6)}')
    
    def get_network_information(self) -> None:
        # Provides a summary of the network parameters. 
        for i, k in enumerate(self.hidden_layers):
            print(f'Type: {type(k).__name__} {i+1} |Shape: {k.weights.shape}\t | Parameters: {k.weights.shape[0] * k.weights.shape[0] + k.biases.shape[0] * k.biases.shape[1]}\t | Activation: {type(k.activation).__name__}')
        print(f'Loss function: {type(self.loss_function).__name__}')
        print(f'Metric: {type(self.metric).__name__}')
                 
    def save_model(self,file) -> None:
        with open(file,'wb') as file:
            pickle.dump(self,file)
            
    def load_model(file):
        with open(file,'rb') as file:
            return  pickle.load(file)