import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib
from keras.utils import multi_gpu_model
from os.path import join

from refactored.classification.finetuning.time_history import TimeHistory
from refactored.preprocessing.property.illumination_extractor import IlluminationExtractor
from refactored.preprocessing.property.original_extractor import OriginalExtractor

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from typing import List, Tuple

import numpy as np
import os
from keras import Model
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.layers import Dense
from keras.optimizers import Adam

from refactored.feature_extraction.cnn_model import CnnModel
from refactored.io_utils import save_txt
from refactored.preprocessing.handler.datahandler import DiskHandler, DataHandler
from refactored.preprocessing.property.property_extractor import PropertyExtractor
from tools.classifier import evaluate_hter
from tools.file_utils import file_helper
import pickle


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class BaseFinetuner():
    PRED_NAME = "y_pred.npy"
    PROBA_NAME = "y_pred_proba.npy"
    MODEL_NAME = "model.h5"
    WEIGHTS_NAME = "weights.h5"
    RESULTS_NAME = "results.txt"
    HISTORY_NAME = "history.pickle"
    TIMES_NAME = "times.pickle"

    IMG_LOSS_EPOCH = "loss_epoch.png"
    IMG_ACC_EPOCH = "acc_epoch.png"

    INTRA_NAME = "intra"
    INTER_NAME = "inter"
    META_NAME = "meta"

    BATCH_SIZE = 16

    exts = ('*.jpg', '*.png')
    frame_delimiter = '_frame_'

    def __init__(self,
                 images_root_path: str,
                 base_output_path: str,
                 properties: List[PropertyExtractor],
                 models: List[CnnModel],
                 data_handler: DataHandler = DiskHandler(),
                 train_alias: str = 'train',
                 test_alias: str = 'test',
                 target_all: str = 'all'):

        self.images_root_path = images_root_path
        self.base_output_path = base_output_path
        self.properties = properties
        self.models = models
        self.data_handler = data_handler
        self.train_alias = train_alias
        self.test_alias = test_alias
        self.target_all = target_all

        self.intra_dataset_output = os.path.join(self.base_output_path, self.INTRA_NAME)
        self.inter_dataset_output = os.path.join(self.base_output_path, self.INTER_NAME)
        self.meta_dataset_output = os.path.join(self.base_output_path, self.META_NAME)

    def _list_variations(self):

        # models = [resnet, vgg, etc]
        for model in self.models:
            for prop in self.properties:
                yield [model, prop]

    def _save_artifacts(self, model: CnnModel,
                        history,
                        output_dir: str,
                        y_pred: np.ndarray,
                        results: np.ndarray,
                        time_callback):

        file_helper.guarantee_path_preconditions(output_dir)

        # save preds
        np.save(os.path.join(output_dir, self.PRED_NAME), y_pred)

        # save HTER, APCER and BPCER
        results_path = os.path.join(output_dir, self.RESULTS_NAME)
        result = '%.5f\n%.5f\n%.5f' % (results[0], results[1], results[2])
        print('results:', results)
        print('result str:', result)
        save_txt(results_path, result)

        model.save(join(output_dir, self.MODEL_NAME))
        model.save_weights(join(output_dir, self.WEIGHTS_NAME))

        with open(join(output_dir, self.HISTORY_NAME), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        with open(join(output_dir, self.TIMES_NAME), 'wb') as file_pi:
            pickle.dump(time_callback.times, file_pi)

        plt.clf()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(join(output_dir, self.IMG_ACC_EPOCH))
        plt.clf()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig(join(output_dir, self.IMG_LOSS_EPOCH))

    def train(self, train_batches, test_batches, model, num_train_steps, num_test_steps):
        classes = ["attack", "real"]

        model.layers.pop()

        for layer in model.layers:
            layer.trainable = False

        last = model.layers[-1].output

        classification_layer = Dense(len(classes), activation="softmax")(last)

        ft_model = Model(model.input, classification_layer)
        # ft_model = multi_gpu_model(ft_model, gpus=3)
        ft_model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

        time_callback = TimeHistory()

        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        #                               patience=5, min_lr=0.00001)

        tensorboard = TensorBoard(log_dir='/codes/bresan/remote/spoopy/spoopy/refactored/classification/finetuning',
                                  histogram_freq=0,
                                  write_graph=True, write_images=False)

        history = ft_model.fit_generator(train_batches,
                                         steps_per_epoch=num_train_steps,
                                         epochs=50,
                                         callbacks=[time_callback],
                                         validation_data=test_batches,
                                         validation_steps=num_test_steps)

        return ft_model, history, time_callback

    def _fit(self, model: CnnModel,
             X_train: np.ndarray,
             y_train: np.ndarray) -> CnnModel:
        X_train = np.reshape(X_train, (X_train.shape[0], -1))

        model.fit(X_train, y_train)
        return model

    def _predict(self, model: CnnModel,
                 X_test: np.ndarray) -> np.ndarray:
        y_pred = model.predict(X_test)
        return y_pred

    def _evaluate_results(self, y_pred, y_test, names_test) -> Tuple[float, float, float]:

        y_test = y_test[:, 1]  # single column for being a real video. 0 = attack, 1 = real
        y_pred = y_pred[:, 1]  # real column

        for i, pred in enumerate(y_pred):
            if pred > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        hter, apcer, bpcer = evaluate_hter.evaluate_with_values(y_pred, y_test, names_test)
        return hter, apcer, bpcer
