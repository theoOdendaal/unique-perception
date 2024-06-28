import numpy as np

class PreprocessData(object):
    '''Preprocess the training and testing data'''
    def __init__(
        self,
        train_input,
        train_output,
        test_input,
        test_output,
        output_count,
        sample_size,
        batch_size,
        dimensions = 2
        ) -> None:
        
        self.output_count = output_count
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.dimensions = dimensions
        
        '''Formats the populations'''
        self.X_pop, self.Y_pop = self.add_population(train_input, train_output)
        self.x_pop, self.y_pop = self.add_population(test_input, test_output)
        
        '''Creates a sample from the population'''
        self.X_sample, self.Y_sample = self.create_sample(self.X_pop,self.Y_pop,self.sample_size)
        self.x_sample, self.x_sample = self.create_sample(self.x_pop,self.y_pop,self.sample_size)
        
        '''Converts sample into batches'''
        self.X_batch, self.Y_batch = self.create_batch(self.X_sample,self.Y_sample, self.batch_size)
        
        '''Sets attributes used by Neural Network'''
        self.input_size = len(self.X_pop[0])
         
    def add_population(self, x_data, y_data):
        return (self.Input(x_data, self.dimensions).data, self.Output(y_data,self.output_count).data)
           
    def create_sample(self, x_d, y_d, sample_size):
        if len(x_d) != len(y_d):
            return 'Matching input and output data have not been provided.'
        del_range = range(np.max(len(x_d)-sample_size,0))
        return np.delete(x_d, del_range,axis=0), np.delete(y_d, del_range,axis=0)  

    def create_batch(self, x_d, y_d, batch_size):
        uneven = len(x_d) % batch_size
        x = np.split(x_d[uneven:], len(x_d[uneven:])/batch_size, axis=0)
        y = np.split(y_d[uneven:], len(y_d[uneven:])/batch_size, axis=0)
        x.append(x_d[:uneven])
        y.append(y_d[:uneven])
        return x, y 
    
    
    class Input(object):
        '''Input data subclass'''
        def __init__(self, data, dim):
            self.data = self.fit_network_parameters(dim, data).astype('float32')
        
        def fit_network_parameters(self,dim, d):
            if len(d.shape) > dim:
                size = 1
                for k in range(1,len(d.shape)):
                    size *= d.shape[k]
                return d.reshape(d.shape[0],size)
            return d
                         
    class Output(object):
        '''Output data subclass'''
        def __init__(self, data, output):
            self.data = self.hot_encode(self.create_array(data),output).astype('float32')
                        
        def create_array(self,d):
            if len(d.shape) == 1:
                return d.reshape(len(d),1)
            else:
                return d
        
        def hot_encode(self, d, output):
            if len(d.shape) > 1:
                if d.shape[1] != output:
                    temp = np.zeros((len(d),output)) 
                    for i,k in enumerate(d):
                        temp[i,k] = 1
            return temp
    