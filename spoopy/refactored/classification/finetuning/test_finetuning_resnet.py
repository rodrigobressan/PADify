import matplotlib

matplotlib.use('Agg')

import json

import keras
import math
import os
from PIL import ImageFile
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam

ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt

# DATA_DIR = '/codes/bresan/remote/spoopy/spoopy/refactored/tests/artifacts_bkp/extracted_frames/cbsr'
TRAIN_DIR = '/codes/bresan/remote/spoopy/spoopy/refactored/tests/artifacts_bkp/aligned/cbsr/train/attack/original'
VALID_DIR = '/codes/bresan/remote/spoopy/spoopy/refactored/tests/artifacts_bkp/aligned/cbsr/test/attack/original'
SIZE = (224, 224)
BATCH_SIZE = 16

# definine the total number of epochs to train for along with the
# initial learning rate
NUM_EPOCHS = 70
INIT_LR = 5e-3


def perform_finetuning():
    model = keras.applications.resnet50.ResNet50()

    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples / BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples / BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator()

    train_data = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='binary', shuffle=True,
                                         batch_size=BATCH_SIZE)

    test_data = gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='binary', shuffle=True,
                                        batch_size=BATCH_SIZE)

    train(train_data, test_data, model, num_train_steps, num_valid_steps)


def train(train_batches, test_batches, model, num_train_steps, num_test_steps):
    classes = list(iter(train_batches.class_indices))
    model.layers.pop()
    for layer in model.layers:
        layer.trainable = False
    last = model.layers[-1].output
    x = Dense(len(classes), activation="softmax")(last)
    finetuned_model = Model(model.input, x)
    # finetuned_model = multi_gpu_model(finetuned_model, gpus=2)
    finetuned_model.compile(optimizer=Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
    for c in train_batches.class_indices:
        classes[train_batches.class_indices[c]] = c
    finetuned_model.classes = classes
    early_stopping = EarlyStopping(patience=10)
    checkpointer = ModelCheckpoint('resnet50_best.h5', verbose=1, save_best_only=True)
    history = finetuned_model.fit_generator(train_batches, steps_per_epoch=num_train_steps, epochs=1000,
                                            callbacks=[early_stopping, checkpointer], validation_data=test_batches,
                                            validation_steps=num_test_steps)
    finetuned_model.save('resnet50_final.h5')
    with open('history.json', 'w') as f:
        json.dump(history.history, f)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc_epoch.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('loss_epoch.png')


if __name__ == "__main__":
    perform_finetuning()
