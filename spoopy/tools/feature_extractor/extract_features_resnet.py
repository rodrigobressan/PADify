import glob
import os
import os.path

import numpy as np

np.random.seed(1)

from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input


def extract_features(path_frames, output_path):
    no_imgs = []  # No. of images

    images = len(glob.glob(os.path.join(path_frames, '*.jpg')))  # assuming the images are stored as 'jpg'
    no_imgs.append(images)
    num_samples = np.sum(no_imgs)  # total number of all samples

    # Compute the features
    width, height, channels = (224, 224, 3)
    X = np.zeros((num_samples, width, height, channels))
    cnt = 0
    list_paths = []  # List of image paths
    samples_names = []
    print("Processing images ...")
    for img_file in glob.glob(path_frames + '/*.jpg'):
        # print("[%d] Processing image: %s" % (cnt, img_file))
        list_paths.append(os.path.join(os.getcwd(), img_file))
        img = image.load_img(img_file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X[cnt] = x
        cnt += 1
        samples_names.append(img_file)

    print("Images processed: %d" % (cnt))

    # Creating base_model (ResNet50 notop)
    image_shape = (224, 224, 3)

    from keras import backend
    with backend.get_session().graph.as_default() as g:
        base_model = ResNet50(weights='imagenet', input_shape=image_shape, include_top=False)

        filename = os.path.join(output_path, 'features_resnet.npy')
        resnet50features = base_model.predict(X)

        np.save(filename, resnet50features)
    # print('featurs shape: ', resnet50features.shape)



if __name__ == '__main__':

    base_path = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/static/client/frames/bafe8c3d6cfb99a812389a5a57fd7b47_mp4/saliency'
    output_path = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/static/client/frames/bafe8c3d6cfb99a812389a5a57fd7b47_mp4/saliency'
    extract_features(base_path, output_path)
