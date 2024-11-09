'''The purpose of this exercice is to make the dataset data/aquarium usable.

First, unzip the archive data/aquarium.zip.

Then, fill the data.yaml file (you dont need to do this automatically).

After that, you'll have to separate it in 3 sets (train, validation, test) randomly
and save each image and target in it's set directory in (data/aquarium). Look
at the data/mnist dataset if you need example. (This part must be done 
automatically in this file).

To verify if it works, run the function train_yolo.

Then load each part of the dataset in a numpy array (with normalization and 
vectorized images).
'''

from ultralytics import YOLO

AQUARIUM_YAML_PATH = 'data/aquarium/data.yaml'

def train_yolo():
    model = YOLO('model/yolov8n.pt')
    model.train(data=AQUARIUM_YAML_PATH)

def make_dataset():
    pass

def load_dataset():
    pass

def main():
    make_dataset()
    # train_yolo()
    load_dataset()

if __name__ == '__main__':
    main()