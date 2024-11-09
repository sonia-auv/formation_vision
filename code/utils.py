import os
import cv2
from sklearn import datasets
from sklearn.model_selection import train_test_split

MNIST_DIR = './data/mnist/'
IMG_DIR = 'image/'
TARGET_DIR = 'target/'
TRAIN_DIR = 'train/'
VAL_DIR = 'val/'
TEST_DIR = 'test/'

YAML_PATH = './data/mnist/data.yaml'
YAML_CONTENT = 'path: ./data/mnist\n\ntrain: train/image\nval: val/image\ntest: test/image\n\nnames: \n  0: 0\n  1: 1\n  2: 2\n  3: 3\n  4: 4\n  5: 5\n  6: 6\n  7: 7\n  8: 8\n  9: 9'

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def download_mnist_dataset():
    digits = datasets.load_digits()

    '''Splitting the dataset in 3 parts: 
        - train = 70%,
        - val = 20%,
        - test = 10%.
    Proportions might change a bit but you should stay around these values.'''
    x_train, x_val, y_train, y_val = train_test_split(digits.images, digits.target, train_size=.70)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, train_size=2/3)

    x = [x_train, x_val, x_test]
    y = [y_train, y_val, y_test]
    dirs = [TRAIN_DIR, VAL_DIR, TEST_DIR]
    return x, y, dirs

def save_mnist_dataset(x, y, dirs):
    for k, dir in enumerate(dirs):
        make_dir(MNIST_DIR+dir)
        make_dir(MNIST_DIR+dir+IMG_DIR)
        make_dir(MNIST_DIR+dir+TARGET_DIR)
        for i, img in enumerate(x[k]):
            img_path = f'{MNIST_DIR}{dir}{IMG_DIR}digit_{i}.jpg'
            target_path = f'{MNIST_DIR}{dir}{TARGET_DIR}digit_{i}.txt'
            cv2.imwrite(img_path, img)
            with open(target_path, 'w+') as f:
                f.write(str(y[k][i]))

# Initialization of the dataset (an extract of MNIST dataset)
def init_mnist():
    if not os.path.exists(MNIST_DIR):
        print('Creating Mnist dataset')
        make_dir(MNIST_DIR)
        with open(YAML_PATH, 'w+', encoding='utf-8') as f:
            f.write(YAML_CONTENT)
        x, y, dirs = download_mnist_dataset()
        save_mnist_dataset(x, y, dirs)