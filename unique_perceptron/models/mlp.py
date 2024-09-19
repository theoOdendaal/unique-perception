import numpy as np
import pickle

from layers.layers import HiddenLayer
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
        
    def __str__(self) -> str:
        output = "\n"
        for index, layer in enumerate(self.hidden_layers):
            output += f"{type(layer).__name__} {index+1}\n"
            output += f"Dimensions:\t{layer.weights.shape}\n"
            output += f"Parameters:\t{layer.weights.shape[0] * layer.weights.shape[0] + layer.biases.shape[0] * layer.biases.shape[1]}\n"
            output += f"Activation:\t{type(layer.activation).__name__}\n\n"
        output += f"Loss function:\t{type(self.loss_function).__name__}\n"
        output += f"Metric:\t{type(self.metric).__name__}\n"
        return output
        
        
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
        for layer in self.hidden_layers:
            self.predicted_output = layer.forward_propagate(self.predicted_output)
        return self.predicted_output
    
    
    def backward_propagate(self) -> None:
        # Calculate initial error gradient
        #self.dz = self.predicted_output - self.actual_output
        self.dz = self.loss_function.cost_derivative(self.actual_output, self.predicted_output)
        
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            layer = self.hidden_layers[i]
            
            if i < len(self.hidden_layers) - 1:
                next_layer = self.hidden_layers[i + 1]
                self.dz = np.dot(self.dz, next_layer.weights.T) * layer.activation.backward(layer.z)
            
            self.dw = np.dot(layer.x.T, self.dz)
            self.db = np.sum(self.dz, axis=0, keepdims=True)
        layer.weights -= self.learning_rate * self.dw
        layer.biases -= self.learning_rate * self.db
        """
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            layer = self.hidden_layers[i]
            
            #
            self.dz = layer.activation.backward(layer.a)
            #
            
            self.dw = np.dot(layer.x.T, self.dz)
            self.db = np.sum(self.dz, axis=0, keepdims=True)
            self.dz = np.dot(self.dz, layer.weights.T)# * layer.activation.backward(layer.z)
        layer.weights -= self.learning_rate * self.dw
        layer.biases -= self.learning_rate * self.db
        """
     
       
    def fit_model(self) -> None:
        self.forward_propagate()
        self.backward_propagate()
        
  
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
        self.loss_function.cost(self.actual_output,self.predicted_output)
        self.metric.evaluate(self.get_prediction(self.actual_output), self.get_prediction(self.predicted_output)) 
        return (f'Loss: {self.loss_function.loss:.4f}\t | Confidence: {np.mean(self.get_confidence()):.4f}\t | {type(self.metric).__name__}: {self.metric.get():.2f}')
    
    