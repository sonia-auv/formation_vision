# Importation des librairies nécessaire pour ce fichier
import numpy as np
import cv2
import os
import yaml

class DataloaderExample():
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path
        self.path = None
        self.train_path = None
        self.val_path = None
        self.test_path = None
        # self.label_names is not used in this example.
        self.label_names = None

    '''Loading each part of the dataset and normalizing them if needed.'''
    def load_mnist(self, normalize=False):
        self.__load_yaml()
        x_train, y_train = self.__load_dataset(self.train_path)
        x_val, y_val = self.__load_dataset(self.val_path)
        x_test, y_test = self.__load_dataset(self.test_path)
        if normalize:
            mean = x_train.mean()
            var = x_train.var()
            x_train = (x_train - mean) / var
            x_val = (x_val - mean) / var
            x_test = (x_test - mean) / var
        return x_train, y_train, x_val, y_val, x_test, y_test

    '''Load the information in the yaml file and put it in self variable.'''
    def __load_yaml(self):
        with open(self.yaml_path, 'r+') as f:
            content = yaml.safe_load(f)
        if content['path'] != None:
            self.path = content['path']
            if content['train'] != None:
                self.train_path = content['train']
            if content['val'] != None:
                self.val_path = content['val']
            if content['test'] != None:
                self.test_path = content['test']
            if content['names'] != None:
                self.label_names = content['names']

    '''Load only one part of the dataset.'''
    def __load_dataset(self, data_path):
        '''Setting path variables to the image and target dir.'''
        img_path = self.path  + '/' + data_path + '/'
        target_path = img_path.replace('image', 'target')
        '''Getting all image files name in an array.'''
        img_names = [f for f in os.listdir(img_path) 
                     if os.path.isfile(os.path.join(img_path, f))]
        x = []
        y = []
        '''Iterating on each image file name:'''
        for img_name in img_names:
            '''Getting the target file name corresponding with the image.'''
            target_name = img_name.replace('.jpg', '.txt')
            '''Loading the image as a numpy 2D matrix, 0 means black and white.'''
            img = cv2.imread(img_path + img_name, 0)
            '''Converting the 2D matrix in 1D array.'''
            vect_img = img.ravel()
            '''Getting the target value.'''
            with open(target_path + target_name, 'r+') as f:
                target = int(f.read())
            '''Completing x and y dataset arrays.'''
            x.append(vect_img)
            y.append(target)
        return np.array(x), np.array(y)