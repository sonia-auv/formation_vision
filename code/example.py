'''Introduction to machine learning (here we're not using neural networks
so we don't do deep learning)'''

'''Importing numpy library an renaming it np since it's easier the write'''
import numpy as np

'''Importing pickle library'''
import pickle

'''Importing the function time from time library'''
from time import time 

'''Importing the class SVC (our model's class) from sklearn.svm library'''
from sklearn.svm import SVC

'''Importing the funciton accuracy_score from sklearn.metrics library'''
from sklearn.metrics import accuracy_score

'''Importing the class DataloaderExample from the file code/dataloader.py'''
from dataloader import DataloaderExample

'''Importing the function init_mnist from the file code/utils.py'''
from utils import init_mnist

'''Definition of the path of mnist yaml config file as a constant.'''
MNIST_YAML_PATH = './data/mnist/data.yaml'

def main():
    '''Initialize Mnist dataset by downloading the images (this function is 
    defined in code/utils.py). Only used to initialize this example but still
    interesting, particularly for the separation of the dataset in train, val
    and test parts.'''
    init_mnist()

    '''Declare the object dataset with the class DataloaderExample by giving 
    the path to the yaml configuration file of the Mnist dataset. You should
    look at the file code/dataloader.py where this class if defined.'''
    dataset = DataloaderExample(MNIST_YAML_PATH)
    
    '''Load the dataset in following variables using the method load_mnist()
    from the class DataloaderExample. Even if it looks easy here, its the
    most difficult part in this example. The purpose of this example is to
    give an example of how to load data from an organized dataset. About the
    normalize parameter, if True: data is normalized, if False: it's not.'''
    x_train, y_train, x_val, y_val, x_test, y_test = dataset.load_mnist(normalize=True)
    
    '''Here we concatenate the val and test parts since we dont need the
    validation part yet. We're gonna use it soon but not in this example :)'''
    x_test = np.concatenate((x_test, x_val))
    y_test = np.concatenate((y_test, y_val))
    
    '''Declare our model, its a classifier SVM with 2 hyper-parameters C and kernel
    changing there values may change the results but this example's purpose isn't
    to explain this hyper-parameters so let's that aside for now.'''
    model = SVC(C=1.0, kernel='rbf')

    '''Measuring time before starting to train our model.'''
    t = time()

    '''Training of the model (yes, it's easy in this example).'''
    model.fit(x_train, y_train)

    '''Printing the training time.'''
    print('Training time: {:.2f} secondes'.format(time()-t))

    '''Predicting the results on our test dataset using the trained model.'''
    y_pred = model.predict(x_test)

    '''Measuring the precision by comparing our prediction and the labels
    of the test dataset.'''
    score = accuracy_score(y_test, y_pred)
    print('Precision = {:.2f}%'.format(score*100))

    '''Saving the model in the file model/my_model.pkl'''
    with open('model/my_model', 'wb') as f:
        pickle.dump(model, f)

    '''Loading the model from the file model/my_model.pkl'''
    with open('model/my_model', 'rb') as f:
        loaded_model = pickle.load(f)

if __name__ == '__main__':
    main()