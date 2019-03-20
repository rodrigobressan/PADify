import time
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from sklearn.svm import SVC


def predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x) # extracted feature

    return preds.shape


def test_model(pictures_path):
    print('Current path: ', pictures_path)
    types = os.listdir(pictures_path)
    for type in types:
        print('Current type: ', type)
        frames_path = os.path.join(pictures_path, type)
        frames = os.listdir(frames_path)
        for frame in frames:
            path_frame = os.path.join(frames_path, frame)
            preds = predict(path_frame, model)
            print(preds)


if __name__ == '__main__':
    model_path = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/tools/classifier/resnet50_only_weights.h5'
    print('Loading model:', model_path)
    t0 = time.time()
    model = load_model(model_path)
    t1 = time.time()
    print('Loaded in:', t1 - t0)

    path = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/tools/classifier/flowers/raw-data/validation'
    test_model(path)

    path = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/tools/classifier/flowers/raw-data/train'
    test_model(path)

