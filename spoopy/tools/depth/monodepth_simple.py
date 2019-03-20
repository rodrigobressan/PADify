# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

import concurrent

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.misc
import tensorflow as tf
# only keep warnings and errors
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
from keras import backend as K
from sklearn.externals.six.moves import xrange

from refactored.preprocessing.util.preprocessing_utils import get_not_processed_frames
from tools.depth.monodepth_model import MonodepthModel, monodepth_parameters
from tools.file_utils import file_helper

dirname = os.path.dirname(__file__)

CHECKPOINT_PATH = os.path.join(dirname, 'model/model_cityscapes.data-00000-of-00001')


def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def generate_default_monodepth_parameters():
    """
    This method is used to generate the default parameters used in the monodepth model
    :return:
    """
    params = monodepth_parameters(
        encoder='vgg',
        height=256,
        width=512,
        batch_size=2,
        num_threads=1,
        num_epochs=1,
        do_stereo=False,
        wrap_mode="border",
        use_deconv=False,
        alpha_image_loss=0,
        disp_gradient_loss_weight=0,
        lr_loss_weight=0,
        full_summary=False)

    return params


def chunks(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in xrange(0, len(l), n))


def apply_depth_inference_on_folder(folder_images_path, output_path):
    """
    This method is used to apply depth inference on a given folder (pretty much self explanatory name, right?)
    :param folder_images_path: where our images are located
    :param output_path: where the results of our depth inference will be saved
    :param checkpoint_path: the checkpoint used to perform the inference
    :return:
    """
    params = generate_default_monodepth_parameters()

    output_path = file_helper.guarantee_path_preconditions(output_path)

    left, model, sess = init_tensorflow(CHECKPOINT_PATH, params)

    # all_images = file_helper.get_frames_from_folder(folder_images_path)

    missing_frames = get_not_processed_frames(folder_images_path, output_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        fs = [executor.submit(apply_inference, folder_images_path, image, left, model, output_path,
                              sess) for image in missing_frames]
        concurrent.futures.wait(fs)

    K.clear_session()

    return file_helper.get_frames_from_folder(output_path)


def get_final_name_path(output_path: str, image_name: str) -> str:
    file_name_without_extension = os.path.splitext(image_name)[0]
    file_disp_path = os.path.join(output_path, file_name_without_extension + ".jpg")
    return file_disp_path


def apply_inference(folder_images_path: str, img_name: str, left, model, output_path, sess):
    file_name_path = get_final_name_path(output_path, img_name)

    if os.path.exists(file_name_path):
        print("Depth already done for frame %s" % file_name_path)
        return

    print('calling depth on %s' % img_name)

    input_image = scipy.misc.imread(os.path.join(folder_images_path, img_name))

    original_height, original_width, num_channels = input_image.shape

    input_image = scipy.misc.imresize(input_image, [256, 512])
    input_image = input_image.astype(np.float32) / 255
    stack_images = np.stack((input_image, np.fliplr(input_image)), 0)

    disp = sess.run(model.disp_left_est[0], feed_dict={left: stack_images})
    disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

    disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])

    # and our inferred image
    plt.imsave(file_name_path, disp_to_img)
    print('depth done')


def init_tensorflow(checkpoint_path, params):
    """
    This method is used to initialize Tensorflow, as well to inti our MonodepthModel object
    :param checkpoint_path: the file used as checkpoint
    :param params: the monodepth params
    :return: the left tensorflow placeholder, our model (MonodepthModel) and our sess
    """

    tf.reset_default_graph()
    left = tf.placeholder(tf.float32, [2, 256, 512, 3])
    model = MonodepthModel(params, "test", left, None)

    # init our TensorFlow session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # saver
    train_saver = tf.train.Saver()

    # init global and local variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # restore
    restore_path = checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)
    return left, model, sess


def main():
    apply_depth_inference_on_folder('/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/tools/depth/images',
                                    '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/tools/depth/output')
    print("monodepth_simple main")


if __name__ == '__main__':
    main()
