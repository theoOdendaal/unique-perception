import os
import matplotlib.pyplot as plt
import random
import glob
import ntpath

# Custom modules.
from utils.preprocess import PreprocessData
from layers.core import HiddenLayer
from layers.activation import ReLu, LeakyReLu, Sigmoid, SoftMax
from loss import CategoricalCrossEntropy
from metrics import Accuracy
from models.mlp import MultiLayerPerceptron
    
    
def create_standardized_model(input_size,hidden_layer_count,neurons,output_size,learning_rate,metric,loss_function):
    '''Initializes a standardized MultiLayerPerceptron object instance'''
    '''The following specifications are standardized:
        1. Each HiddenLayer has the same number of neurons (excluding the output layer);
        2. Each HiddenLayer has the same activation function (excluding the output layer);'''
        
    temp = MultiLayerPerceptron(learning_rate)
    size = input_size
    for i in range(hidden_layer_count):
        temp.add_dense(HiddenLayer.initialize_weights(size,neurons),HiddenLayer.initialize_biases(neurons),LeakyReLu)
        size = neurons
    temp.add_dense(HiddenLayer.initialize_weights(size,output_size),HiddenLayer.initialize_biases(output_size),SoftMax)
    return temp

if __name__ == '__main__':
    
    '''SETTINGS'''
    fit_model = True

    '''DIRECTORY'''
    file_directory = os.path.join(os.path.dirname(os.path.realpath(__file__))) 


    '''REQUIRED INPUTS'''
    '''Training setting'''
    SAMPLE_SIZE = 6000 #Number of items used to train neural network.
    BATCH_SIZE = 32 #Number of items contained within a single batch of training data.
    TRAINING_ITERATIONS = 100 #Number of iterations per batch used for purpose of training the neural network.
    UPDATE_FREQUENCY = 1 #Frequency of training process progress being printed.
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

    '''NEURAL NETWORK INITIALIZATION.'''
    if input('Would you like to load a pre-existing model? (Y/N)' + '\n').lower() == 'y': 
        try: #UPDATE THIS TRY EXCEPT STATEMENT!!!!!!!!!!!!!!!!!!!!! AS CURRENTLY THE EXCEPT IS TO GENERIC!!!!!!!!!!!!!!!!!!!!!!
            files = glob.glob(os.path.join(file_directory,'*.pickle'))
            print('The following files are available for import:')
            for i,file in enumerate(files):
                print(f'Name: {ntpath.basename(file)} \t |Reference: {i}')
            
            index = ''
            while index == '':
                index = input("\n" + "Please specify the pickle file you'd like to import?" + "\n")
            network = MultiLayerPerceptron.load_model(os.path.join(file_directory,ntpath.basename(files[int(index)])))
            print('Model successfully loaded.' + '\n')
        except:
            print('Failed to load model.')
    else:
        network = create_standardized_model(data.input_size,hidden_layer_count,neurons,output_size,0.01,Accuracy,CategoricalCrossEntropy)
    network.get_network_information() #Returns network specifications.
    print('\n')


    '''TRAIN NEURAL NETWORK'''
    if fit_model == True:
        print('Training neural network...')
        network.optimize_model(data.X_batch, data.Y_batch, TRAINING_ITERATIONS, UPDATE_FREQUENCY, TRAINING_CYCLES, batch=True)
        print('\n')


    '''EVALUATE MODEL'''
    '''extrapolates the trained neural network over the training data population'''
    print('Training dataset results...')
    network.evaluate(data.X_pop,data.Y_pop)
    print('\n')

    '''extrapolates the trained neural network over the testing data population'''
    print('Testing datasets results...')
    network.evaluate(data.x_pop,data.y_pop)
    print('\n')

    '''SAVE NEURAL NETWORK PARAMETERS'''
    if input('Would you like to save this model ? (Y/N)' + '\n').lower() == 'y':
        try:
            name = ''
            while name == '':
                name = input('File name ?' + '\n')        
            network.save_model(os.path.join(file_directory,f'{name}.pickle'))
            print('Model successfully exported.')
        except:
            print('Failed to export model.')
    print('Script terminated.')

    while True:
        random_selection = random.randrange(0,test_x.shape[0])
        network.set_data(data.x_pop[random_selection],data.y_pop[random_selection])
        print(network.get_prediction(network.forward_propagate()))
        plt.imshow(test_x[random_selection])
        plt.show()
        if input("Would you like to run another test? [y/n]: ").lower() != "y":
            break
            





        
    