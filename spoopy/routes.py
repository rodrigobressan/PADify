import hashlib
import os
import time
from statistics import mean
from threading import Thread

import cv2
import imutils
import numpy as np
from flask import Flask, render_template, make_response
from flask_restful import Api, Resource, reqparse
from sklearn.externals import joblib
from werkzeug.datastructures import FileStorage

from tools import file_utils, feature_extractor, face_aligner
# constants
from tools.depth import monodepth_simple
from tools.face_detector import face_detector
from tools.map_extractor.utils.ImageAligner import ImageAligner
from tools.map_extractor.utils.ImageCropper import ImageCropper
from tools.saliency_extractor import saliency
from tools.vole import predict_illuminant

UPLOAD_FOLDER_KEY = 'UPLOAD_FOLDER'
ARGUMENT_FILE_PARAMETER = 'file'
MYDIR = os.path.join(os.path.dirname(__file__), 'static', 'client')

PATH_VIDEOS = os.path.join(MYDIR, 'videos')
PATH_FRAMES = os.path.join(MYDIR, 'frames')
PATH_FEATURES = os.path.join(MYDIR, 'features')
PATH_PREDICTIONS = os.path.join(MYDIR, 'predictions')

DEPTH_FRAMES = os.path.join(MYDIR, 'depth')
SALIENCY_FRAMES = os.path.join(MYDIR, 'saliency')
ILLUMINATION_FRAMES = os.path.join(MYDIR, 'illumination')

# set up Flask application
app = Flask(__name__)
app.config[UPLOAD_FOLDER_KEY] = 'static/uploads'
api = Api(app)

DEPTH_MODEL_PATH = '/home/tiagojc/spoopy/data/fitted_model_depth.sav'
ILLUMINATION_MODEL_PATH = '/home/tiagojc/spoopy/data/fitted_model_illumination.sav'
SALIENCY_MODEL_PATH = '/home/tiagojc/spoopy/data/fitted_model_saliency.sav'
# DEPTH_MODEL_PATH = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/static/evaluate/cross_dataset_combinations/ra/cbsr/depth/fitted_model.sav';
# ILLUMINATION_MODEL_PATH = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/static/evaluate/cross_dataset_combinations/ra/cbsr/depth/fitted_model.sav';
# SALIENCY_MODEL_PATH = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/static/evaluate/cross_dataset_combinations/ra/cbsr/depth/fitted_model.sav';
#
model_depth = joblib.load(DEPTH_MODEL_PATH)
model_illumination = joblib.load(ILLUMINATION_MODEL_PATH)
model_saliency = joblib.load(SALIENCY_MODEL_PATH)
#
fc = face_detector.FaceCropper()
detector, fa = face_aligner.align_faces.make_face_aligner()


# Root endpoint
class Root(Resource):
    def __init__(self):
        pass

    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('results.html'), 200, headers)


# Upload video endpoint
def save_raw_artifact():
    parser = reqparse.RequestParser()
    parser.add_argument(ARGUMENT_FILE_PARAMETER, type=FileStorage, location='files')
    args = parser.parse_args()
    file = args[ARGUMENT_FILE_PARAMETER]
    file_name = file.filename
    file_extension = file_name.split(".")[1]
    file_hash = compute_md5(file_name)
    new_file_name = file_hash + "_" + file_extension
    full_file_name = new_file_name + "." + file_extension
    output_path = os.path.join(PATH_VIDEOS, full_file_name)
    file_utils.file_helper.guarantee_path_preconditions(PATH_VIDEOS)
    file.save(output_path)
    return new_file_name


class BaseResponse(object):
    def __init__(self, id):
        self.id = id


class UploadVideo(Resource):
    def post(self):
        file_id = save_raw_artifact()
        frames = split_video_with_id(file_id)
        return {'id': file_id,
                'frames': frames}, 201


class UploadImage(Resource):
    def post(self):
        file_id = save_raw_artifact()
        output_frames = os.path.join(PATH_FRAMES, file_id, 'raw')

        file_utils.file_helper.copy_file(os.path.join(PATH_VIDEOS, get_video_name_from_id(file_id)), output_frames)
        print('output_frames: ', output_frames)
        frames = list_frames_full_path(output_frames)
        print('frames: ', frames)
        return {'id': file_id,
                'frames': frames}, 201


class UploadAndIllumination(Resource):
    def post(self):

        try:
            file_id = save_raw_artifact()
            output_frames = os.path.join(PATH_FRAMES, file_id, 'raw')

            file_utils.file_helper.copy_file(os.path.join(PATH_VIDEOS, get_video_name_from_id(file_id)), output_frames)
            frames_unaligned = generate_illumination_maps(file_id)
            return make_short_results_response(frames_unaligned), 200
        except Exception as e:
            return {'e': e}, 500
        except:
            return {"Erro"}, 200


class UploadAndSaliency(Resource):
    def post(self):

        try:
            file_id = save_raw_artifact()
            output_frames = os.path.join(PATH_FRAMES, file_id, 'raw')

            file_utils.file_helper.copy_file(os.path.join(PATH_VIDEOS, get_video_name_from_id(file_id)), output_frames)
            frames_unaligned = generate_saliency_maps(file_id)
            return make_short_results_response(frames_unaligned), 200
        except Exception as e:
            return {'e': e}, 500
        except:
            return {"Erro"}, 200


class UploadAndDepth(Resource):
    def post(self):
        try:
            file_id = save_raw_artifact()
            output_frames = os.path.join(PATH_FRAMES, file_id, 'raw')

            file_utils.file_helper.copy_file(os.path.join(PATH_VIDEOS, get_video_name_from_id(file_id)), output_frames)
            frames_unaligned = generate_depth_maps(file_id)
            return make_short_results_response(frames_unaligned), 200
        except Exception as e:
            return {'e': e}, 500
        except:
            return {"Erro"}, 200


class DepthInference(Resource):
    def post(self):
        item_id = get_video_id()

        frames_unaligned = generate_depth_maps(item_id)
        # frames_aligned = run_align_images(item_id, 'depth', 'jpg')
        # extract_features('depth_aligned', item_id)
        # prediction = perform_prediction(item_id, 'depth_aligned', model_depth)
        return make_short_results_response(frames_unaligned), 200


class IlluminationInference(Resource):
    def post(self):
        item_id = get_video_id()

        frames_unaligned = generate_illumination_maps(item_id)
        # frames_aligned = run_align_images(item_id, 'illumination', 'png')
        # extract_features('illumination_aligned', item_id)
        # prediction = perform_prediction(item_id, 'illumination_aligned', model_illumination)
        return make_short_results_response(frames_unaligned), 200


class SaliencyInference(Resource):
    def post(self):
        item_id = get_video_id()

        frames_unaligned = generate_saliency_maps(item_id)
        # frames_aligned = run_align_images(item_id, 'saliency', 'jpg')
        # extract_features('saliency_aligned', item_id)
        # prediction = perform_prediction(item_id, 'saliency_aligned', model_saliency)
        return make_short_results_response(frames_unaligned), 200


def make_short_results_response(frames):
    return {'frames': frames}


def make_result_response(frames_unaligned, frames_aligned, prediction):
    return {'frames_unaligned': frames_unaligned,
            'frames_aligned': frames_aligned,
            'prediction': prediction}


def make_frames_response(frames_list):
    return {'frames': frames_list}


def list_frames_full_path(dir):
    return sort(list(get_files_full_path(dir)))


PATH_LOCAL = "/home/tiagojc/spoopy/spoopy/spoopy/"
PATH_REMOTE = "http://cruzeiro.cti.gov.br:5000/"


def get_files_full_path(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield os.path.join(path.replace(PATH_LOCAL, PATH_REMOTE), file)


def list_frames(dir):
    return sort(list(get_files(dir)))


def get_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def sort(list):
    return sorted(list, key=str.lower)


# Split video endpoint with id
def split_video_with_id(item_id):
    video_name = get_video_name_from_id(item_id)
    path_videos = os.path.join(PATH_VIDEOS, video_name)
    output_frames = os.path.join(PATH_FRAMES, item_id, 'raw')

    if os.path.exists(output_frames):
        print("Split already done")
        return list_frames_full_path(output_frames)

    file_utils.file_helper.split_video_into_frames(path_videos, output_frames)

    return list_frames_full_path(output_frames)


# Apply depth inference
def generate_depth_maps(item_id):
    path_frames = os.path.join(PATH_FRAMES, item_id, 'raw')
    output_depth = os.path.join(PATH_FRAMES, item_id, 'depth')

    if os.path.exists(output_depth):
        print("Depth already done")
        # return list_frames_full_path(output_depth)

    file_utils.file_helper.guarantee_path_preconditions(output_depth)
    monodepth_simple.apply_depth_inference_on_folder(path_frames, output_depth)

    return list_frames_full_path(output_depth)


# Apply illumination inference
def generate_illumination_maps(item_id):
    folder = os.path.join(PATH_FRAMES, item_id, 'raw')
    output = os.path.join(PATH_FRAMES, item_id, 'illumination')

    if os.path.exists(output):
        print("Illumination already done")
        # return list_frames_full_path(output)

    file_utils.file_helper.guarantee_path_preconditions(output)
    predict_illuminant.predict_illuminant(folder, output)

    return list_frames_full_path(output)


# Apply saliency inference
def generate_saliency_maps(item_id):
    folder = os.path.join(PATH_FRAMES, item_id, 'raw')
    output = os.path.join(PATH_FRAMES, item_id, 'saliency')

    if os.path.exists(output):
        print("Saliency already done")
        # return list_frames_full_path(output)

    file_utils.file_helper.guarantee_path_preconditions(output)
    saliency.extract_rbd_saliency_folder(folder, output)

    return list_frames_full_path(output)


class AlignImages(Resource):
    def post(self):
        item_id = get_video_id()

        run_align_images(item_id)

        return item_id, 201


def run_align_images(item_id, property, extension_frames):
    raw_frames = os.path.join(PATH_FRAMES, item_id, 'raw')
    threads = []

    for single_frame in list_frames(raw_frames):
        thread = Thread(target=align_single_frame,
                        args=(item_id, single_frame, detector, fa, property, extension_frames))
        threads.append(thread)
        thread.start()

        print('done: ', single_frame)

    for thread in threads:
        thread.join()

    return list_frames_full_path(os.path.join(PATH_FRAMES, item_id, property + '_aligned'))


def align_single_frame(item_name, current_frame_name, detector, fa, property, extension_frames):
    final_aligned_dir = os.path.join(PATH_FRAMES, item_name, property + '_aligned')
    final_aligned_path = os.path.join(final_aligned_dir, current_frame_name)

    if os.path.exists(final_aligned_path):
        return

    if not os.path.exists(final_aligned_dir):
        os.makedirs(final_aligned_dir, exist_ok=True)

    original_frame = os.path.join(PATH_FRAMES, item_name, 'raw', current_frame_name)
    original_angle = face_aligner.align_faces.get_face_angle(original_frame, detector, fa)
    original_rotated = imutils.rotate(cv2.imread(original_frame), original_angle)

    coordinates = fc.get_faces_coordinates(original_rotated)

    print('coordinates none: ', coordinates is None)
    cropper = ImageCropper(fc, coordinates)

    original_path_frame = os.path.join(PATH_FRAMES, item_name, property, current_frame_name)
    aligner = ImageAligner(original_path_frame, original_angle, extension_frames)
    aligned_img = aligner.align()
    print('final: ', final_aligned_path)
    cropper.crop(aligned_img, final_aligned_path)


class FeatureExtractor(Resource):
    def post(self):
        item_id = get_video_id()

        properties = ['depth', 'illumination', 'saliency']
        threads = []
        for property in properties:
            thread = Thread(target=extract_features, args=(property, item_id))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return item_id, 201


def perform_prediction(item_id, type, model):
    features = get_feature(type, item_id)
    output_pred_file = os.path.join(PATH_PREDICTIONS, remove_extension(item_id), type)

    file_utils.file_helper.guarantee_path_preconditions(output_pred_file)
    predictions = model.predict(features)
    predictions_proba = model.predict_proba(features)
    print('predictions: ', predictions)
    print('predictions_proba: ', predictions_proba)
    np.save(os.path.join(output_pred_file, 'pred.npy'), predictions)
    np.save(os.path.join(output_pred_file, 'pred_proba.npy'), predictions_proba)

    return mean(predictions_proba[:, 1].tolist())


def run_predictions(item_id):
    depth_predictions = perform_prediction(item_id, 'depth_aligned', model_depth)
    illumination_predictions = perform_prediction(item_id, 'illumination_aligned', model_illumination)
    saliency_predictions = perform_prediction(item_id, 'saliency_aligned', model_saliency)

    predictions_list = [mean(depth_predictions[:, 1].tolist()),
                        mean(illumination_predictions[:, 1].tolist()),
                        mean(saliency_predictions[:, 1].tolist())]
    return predictions_list


class Predictor(Resource):
    def post(self):
        item_id = get_video_name()

        predictions_list = run_predictions(item_id)

        return predictions_list, 201


class Crash(Resource):
    def get(self):
        raise Exception("Crashing server..")


class Process(Resource):
    def post(self):
        time_begin = time.time()

        # upload video
        file_name = save_raw_artifact()
        time_upload = time.time()
        print('upload done ')

        # split video into frames
        item_id = split_video_with_id(file_name)
        time_split = time.time()
        print('split done')

        # generate depth maps
        generate_depth_maps(item_id)
        time_depth = time.time()
        print('depth done')

        # generate illumination maps
        generate_illumination_maps(item_id)
        time_illumination = time.time()
        print('illumination done')

        # generate saliency maps
        generate_saliency_maps(item_id)
        time_saliency = time.time()
        print('saliency done')

        # align images
        run_align_images(file_name)
        time_align = time.time()
        print('align done')

        # extract features
        extract_all_features(file_name)
        time_features = time.time()
        print('features done')

        # run predictions
        preds = run_predictions(file_name)
        time_predictions = time.time()
        print('predictions done')

        print("Overall results")
        print("Upload time: %.2f" % (time_upload - time_begin))
        print("Split time: %.2f" % (time_split - time_upload))
        print("Depth time: %.2f" % (time_depth - time_split))
        print("Illumination time: %.2f" % (time_illumination - time_depth))
        print("Saliency time: %.2f" % (time_saliency - time_illumination))
        print("Align time: %.2f" % (time_align - time_saliency))
        print("Features time: %.2f" % (time_features - time_align))
        print("Prediction time: %.2f" % (time_predictions - time_features))

        return preds, 201


def perform_all_steps(file_name):
    item_id = split_video_with_id(file_name)
    print('split done')

    generate_depth_maps(item_id)
    print('depth done')
    # run_saliency(item_id)
    run_align_images(file_name)
    print('align done')
    extract_all_features(file_name)
    print('features done')
    preds = run_predictions(file_name)

    print('preds done')


def get_feature(type, item_id):
    path_feature = os.path.join(PATH_FEATURES, remove_extension(item_id), type, 'features_resnet.npy')
    features = np.load(path_feature)
    features = np.reshape(features, (features.shape[0], -1))

    return features


def extract_all_features(item_id):
    extract_features('depth_aligned', item_id)
    extract_features('illumination_aligned', item_id)
    extract_features('saliency_aligned', item_id)


def extract_features(type, item_id):
    folder_depth = os.path.join(PATH_FRAMES, item_id, type)
    folder_features_output = os.path.join(PATH_FEATURES, item_id, type)

    if os.path.exists(folder_features_output):
        return

    file_utils.file_helper.guarantee_path_preconditions(folder_features_output)
    feature_extractor.extract_features_resnet.extract_features(folder_depth, folder_features_output)


def video_exists(id):
    output_path = os.path.join(MYDIR, 'frames', id)
    return os.path.exists(output_path)


def remove_extension(name):
    return name.split(".")[0]


def get_video_name_from_id(video_id):
    name_without_ext = video_id.rsplit(".")[0]
    file_name = name_without_ext.split("_")[0]
    file_ext = name_without_ext.split("_")[1]

    full_name = file_name + "_" + file_ext + "." + file_ext

    return full_name


def get_video_id():
    parser = reqparse.RequestParser()
    parser.add_argument('id', type=str, help='Video.py id')

    args = parser.parse_args()

    id = args['id']
    return id


def get_video_name():
    parser = reqparse.RequestParser()
    parser.add_argument('id', type=str, help='Video.py id')

    args = parser.parse_args()

    name_without_ext = args['id'].rsplit(".")[0]
    file_name = name_without_ext.split("_")[0]
    file_ext = name_without_ext.split("_")[1]

    full_name = file_name + "_" + file_ext + "." + file_ext

    return full_name


def compute_md5(my_string):
    m = hashlib.md5()
    m.update(my_string.encode('utf-8'))
    return m.hexdigest()


api.add_resource(Root, '/')
api.add_resource(UploadVideo, '/upload_video')
api.add_resource(UploadAndDepth, '/upload_depth')
api.add_resource(UploadAndIllumination, '/upload_illumination')
api.add_resource(UploadAndSaliency, '/upload_saliency')
api.add_resource(UploadImage, '/upload_image')
api.add_resource(DepthInference, '/depth')
api.add_resource(IlluminationInference, '/illumination')
api.add_resource(SaliencyInference, '/saliency')
api.add_resource(AlignImages, '/align')
api.add_resource(FeatureExtractor, '/feature_extractor')
api.add_resource(Predictor, '/predictor')
api.add_resource(Process, '/process')
api.add_resource(Crash, '/crash')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    import logging

    logging.basicConfig(filename='error.log', level=logging.DEBUG)
    app.run(host='0.0.0.0', port=port, threaded=True)
