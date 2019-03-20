import math, json, os, sys

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing import image


DATA_DIR = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/tools/classifier/android_sample'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
SIZE = (224, 224)
BATCH_SIZE = 1


if __name__ == "__main__":

    image_shape = (224, 224, 3)
    model = keras.applications.resnet50.ResNet50(input_shape=image_shape, weights='imagenet', include_top=False)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.layers[-1].output)
    model.save('resnet50_only_weights.h5')