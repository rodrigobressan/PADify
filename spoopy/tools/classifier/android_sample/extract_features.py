# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from sklearn.externals import joblib

np.random.seed(1337)  # for reproducibility
from keras.models import load_model

import tensorflow as tf

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def get_img(file_name):
    img = image.load_img(file_name, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def predict_pb(graph, img):
    imTensor = img

    input_height = 224
    input_width = 224
    input_mean = 0
    input_std = 255
    input_layer = "input_1"
    output_layer = "output_node0"
    #
    # t = read_tensor_from_image_file(
    #     file_name,
    #     input_height=input_height,
    #     input_width=input_width,
    #     input_mean=input_mean,
    #     input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results_pb = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: imTensor
        })

    return results_pb


def run_dir(dir):
    print('Current dir: ', dir)
    types = os.listdir(dir)
    for type in types:
        print('Current type: ', type)
        items_train_dir = os.path.join(train_dir, type)
        for item_train in os.listdir(items_train_dir):
            item_path = os.path.join(items_train_dir, item_train)

            print(item_path)

            img = get_img(item_path)

            results_h5 = h5_model.predict(img)
            results_pb = predict_pb(pb_model, img)
            print('H5: ', results_h5)
            print('PB: ', results_pb)

            results_h5 = np.reshape(results_h5, (results_h5.shape[0], -1))

            print("H5 predict: ", svc.predict(results_h5))
            results_pb = np.reshape(results_pb, (results_pb.shape[0], -1))

            print("PB predict: ", svc.predict(results_pb))



if __name__ == "__main__":
    h5_file = "/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/tools/classifier/android_sample/resnet50_only_weights.h5"
    pb_file = "/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/tools/classifier/android_sample/resnet50_only_weights.pb"

    # perform TensorFlow extraction
    pb_model = load_graph(pb_file)
    h5_model = load_model(h5_file)

    svc = joblib.load('fitted_print.sav')

    test_dir = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/tools/classifier/android_sample/test'
    train_dir = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/tools/classifier/android_sample/train'

    run_dir(train_dir)
    run_dir(test_dir)
