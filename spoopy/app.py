import os

from flask import Flask, json, request, render_template, jsonify, Response
from werkzeug.utils import secure_filename

# simple configuration to define where our files will be uploaded
from Dataset import Dataset
from DatasetsReponse import DatasetsResponse
from Type import Type
from tools import file_utils

# basic configuration used on our upload section
allowed_extensions = set(['avi'])
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# where the filed will be uploaded
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/uploads')

# where the results of the inferences are stored
RESULTS_FOLDER = os.path.join(APP_ROOT, 'static/results')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET'])
def render_index():
    """
    This method is used to render the index page of our application (/)
    :return: the index page
    """
    return render_template('results.html')


@app.route('/results', methods=['GET'])
def render_results():
    """
    This method is used to render the results page
    :return: the render template of the results page
    """
    return render_template('results.html')


@app.route('/results/<dataset>/<type>/<id>/compare/<dataset_other>/<type_other>/<id_other>', methods=['GET'])
def render_compare(dataset, type, id, dataset_other, type_other, id_other):
    item_path = os.path.join(dataset, type, id)
    base_url = os.path.join('results', item_path)

    # TODO check for name as 'original'
    original_video_url = os.path.join(base_url, 'video_original.mp4')
    depth_video_url = os.path.join(base_url, 'video_depth.mp4')
    illuminated_video_url = os.path.join(base_url, 'video_illuminated.mp4')
    grayworld_video_url = os.path.join(base_url, 'video_grayworld.mp4')

    average_depth_url = os.path.join(base_url, 'frames', 'average_depth.jpg')
    average_normal_url = os.path.join(base_url, 'frames', 'average_normal.jpg')
    average_grayworld_url = os.path.join(base_url, 'frames', 'average_grayworld.png')
    average_illuminated_url = os.path.join(base_url, 'frames', 'average_illuminated.png')

    other_item_path = os.path.join(dataset, type_other, id_other)
    other_base_url = os.path.join('results', other_item_path)

    # TODO check for name as 'original'
    other_original_video_url = os.path.join(other_base_url, 'video_original.mp4')
    other_depth_video_url = os.path.join(other_base_url, 'video_depth.mp4')
    other_illuminated_video_url = os.path.join(other_base_url, 'video_illuminated.mp4')
    other_grayworld_video_url = os.path.join(other_base_url, 'video_grayworld.mp4')

    other_average_depth_url = os.path.join(other_base_url, 'frames', 'average_depth.jpg')
    other_average_normal_url = os.path.join(other_base_url, 'frames', 'average_normal.jpg')
    other_average_grayworld_url = os.path.join(other_base_url, 'frames', 'average_grayworld.png')
    other_average_illuminated_url = os.path.join(other_base_url, 'frames', 'average_illuminated.png')

    return render_template('compare.html',
                           original_video_url=original_video_url,
                           depth_video_url=depth_video_url,
                           average_depth_url=average_depth_url,
                           average_normal_url=average_normal_url,
                           average_grayworld_url=average_grayworld_url,
                           average_illuminated_url=average_illuminated_url,
                           illuminated_video_url=illuminated_video_url,
                           grayworld_video_url=grayworld_video_url,
                           other_original_video_url=other_original_video_url,
                           other_depth_video_url=other_depth_video_url,
                           other_average_depth_url=other_average_depth_url,
                           other_average_normal_url=other_average_normal_url,
                           other_average_grayworld_url=other_average_grayworld_url,
                           other_average_illuminated_url=other_average_illuminated_url,
                           other_illuminated_video_url=other_illuminated_video_url,
                           other_grayworld_video_url=other_grayworld_video_url)


@app.route('/results/<dataset>/<type>/<id>', methods=['GET'])
def render_item_details(dataset, type, id):
    """
    This method is used to render the details of an already processed item
    :param dataset: the dataset we are looking into (cbsr/nuaa)
    :param type: the type of the data (real/fake)
    :param id: the id of the content we are looking for (1_1, 0001, etc)
    :return: the render template of the item details page
    """
    item_path = os.path.join(dataset, type, id)
    base_url = os.path.join('results', item_path)

    # TODO check for name as 'original'
    original_video_url = os.path.join(base_url, 'video_original.mp4')
    depth_video_url = os.path.join(base_url, 'video_depth.mp4')
    illuminated_video_url = os.path.join(base_url, 'video_illuminated.mp4')
    grayworld_video_url = os.path.join(base_url, 'video_grayworld.mp4')

    average_depth_url = os.path.join(base_url, 'frames', 'average_depth.jpg')
    average_normal_url = os.path.join(base_url, 'frames', 'average_normal.jpg')
    average_grayworld_url = os.path.join(base_url, 'frames', 'average_grayworld.png')
    average_illuminated_url = os.path.join(base_url, 'frames', 'average_illuminated.png')

    return render_template('item.html',
                           path=item_path.replace("/", " > "),
                           original_video_url=original_video_url,
                           depth_video_url=depth_video_url,
                           average_depth_url=average_depth_url,
                           average_normal_url=average_normal_url,
                           average_grayworld_url=average_grayworld_url,
                           average_illuminated_url=average_illuminated_url,
                           illuminated_video_url=illuminated_video_url,
                           grayworld_video_url=grayworld_video_url)


@app.route('/datasets', methods=['GET'])
def retrieve_datasets():
    """
    This method is used to return all the available datasets, as well as its existent types and the files inside of it
    :return: the datasets and their info
    """
    response = DatasetsResponse()
    datasets = {}

    datasets_dirs = file_utils.file_helper.get_dirs_from_folder(RESULTS_FOLDER)

    # go through all our existent datasets
    for dataset_dir in datasets_dirs:
        datasets[dataset_dir] = Dataset(dataset_dir)

        types = {}
        types_dataset = file_utils.file_helper.get_dirs_from_folder(os.path.join(RESULTS_FOLDER, dataset_dir))

        # go through all the existent types
        for type in types_dataset:
            types[type] = Type(type)

            items = file_utils.file_helper.get_dirs_from_folder(os.path.join(RESULTS_FOLDER, dataset_dir, type))

            # go through all the existent items
            for item in items:
                types[type].addItem(item)

            datasets[dataset_dir].addType(types[type])

        response.addDataset(datasets[dataset_dir])

    return Response(json.dumps(response, default=lambda x: x.__dict__), mimetype='application/json')


def is_file_allowed(filename):
    """
    This method is used to check if an uploaded file is valid (for sure we don't want any XSS attacks, right?)
    :param filename: the filename being uploaded
    :return: boolean if the file is valid or not
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/upload', methods=['POST'])
def upload_video():
    """
    This method is used to upload a video into the project. It does add the video in the static/videos folder
    :return: the path where the video was uploaded
    """
    # check if the post request has the file part
    if 'file' not in request.files:
        return json.dumps({'fileName': 'no file'})

    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        return json.dumps({'fileName': 'no_filename'})
    if file and is_file_allowed(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'] + "/video", filename))
        return jsonify(filename=app.config['UPLOAD_FOLDER'] + "/video/" + filename)

    # TODO call monodepth inference here!
    return jsonify(error='error')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
