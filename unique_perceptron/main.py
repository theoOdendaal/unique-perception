import os
import matplotlib.pyplot as plt
import random

# Custom modules.
from utils.preprocess import PreprocessData
from layers.layers import HiddenLayer
from layers.activation import ReLu, LeakyReLu, Sigmoid, SoftMax
from loss import CategoricalCrossEntropy
from metrics import Accuracy
from models.mlp import MultiLayerPerceptron
    
    
def create_standardized_model(input_size, hidden_layer_count, neurons, output_size, learning_rate):
    networkd = MultiLayerPerceptron(learning_rate)
    size = input_size
    for i in range(hidden_layer_count):
        networkd.add_dense(HiddenLayer.initialize_weights(size,neurons), HiddenLayer.initialize_biases(neurons), LeakyReLu)
        size = neurons
    networkd.add_dense(HiddenLayer.initialize_weights(size,output_size), HiddenLayer.initialize_biases(output_size), SoftMax)
    return networkd

def optimize_model(network, input, output, iterations, cycles) -> None:
    network.metric.set_zero()
    for _ in range(cycles):
        for (i, o) in zip(input, output):
            network.set_data(i, o)
            for epoch in range(1,iterations+1):
                network.fit_model()
                #print(network.get_statistics())
    network.metric.set_zero()


if __name__ == '__main__':
    
    '''SETTINGS'''
    fit_model = True

    '''DIRECTORY'''
    file_directory = os.path.join(os.path.dirname(os.path.realpath(__file__))) 


    '''REQUIRED INPUTS'''
    '''Training setting'''
    SAMPLE_SIZE = 60000 #Number of items used to train neural network.
    BATCH_SIZE = 32 #Number of items contained within a single batch of training data.
    TRAINING_ITERATIONS = 50 #Number of iterations per batch used for purpose of training the neural network.
    TRAINING_CYCLES = 1

    '''Hyper parameters'''
    hidden_layer_count = 1 #Number of hidden layers within neural network.
    neurons = 100 #Number of neurons per hidden layer.
    output_size = 10 #Number of neural network output classes.
    learning_rate = 0.01 #Rate at which stochastic gradient descent is applied for purpose of backpropogation.


    '''KERAS DATASETS'''
    from keras.datasets import mnist as dataset
    #from keras.datasets import cifar10 as dataset
    #from keras.datasets import fashion_mnist as dataset

    '''DATA IMPORT AND FORMATTING'''
    (train_x,train_y), (test_x, test_y) = dataset.load_data()
    data = PreprocessData(train_x, train_y, test_x, test_y, output_size, SAMPLE_SIZE, BATCH_SIZE)

    network = create_standardized_model(data.input_size, hidden_layer_count, neurons, output_size, 0.01)
    print(network)

    optimize_model(network, data.X_batch, data.Y_batch, TRAINING_ITERATIONS, TRAINING_CYCLES)
    
    print('\nTraining dataset results...')
    network.evaluate(data.X_pop,data.Y_pop)

    print('\nTesting datasets results...')
    network.evaluate(data.x_pop,data.y_pop)

    while True:
        random_selection = random.randrange(0,test_x.shape[0])
        network.set_data(data.x_pop[random_selection],data.y_pop[random_selection])
        print(network.get_prediction(network.forward_propagate()))
        plt.imshow(test_x[random_selection])
        plt.show()
        if input("Would you like to run another test? [y/n]: ").lower() != "y":
            break
            





        
    