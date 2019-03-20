import os
import sys
from threading import Thread

sys.path.append('/home/carlos/developmen'
                't/spoopy/spoopy')
from tools.file_utils import file_helper

BASE_PATH = '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/evaluate/intra'


def move_files_to_classify(frames_path_origin, type):
    file_helper.get_frames_from_folder(frames_path_origin)


def move_averages_to_classifier_folder(dataset_path):
    ''''
    Required dataset structure:
    <dataset_name>:
        <type_1>:
            <item_1>:
                frames:
                    average_normal.jpg
                    average_depth.jpg
            <item_2>
                frames:
                    average_normal.jpg
                    average_depth.jpg
        <type_2>:
    '''
    types = os.listdir(dataset_path)
    for type in types:
        print(type)

        items = os.listdir(os.path.join(dataset_path, type))
        for item in items:
            # average_illuminant_aligned
            frame_depth = os.path.join(dataset_path, type, item, 'average_depth_aligned.png')
            final_path = os.path.join(
                '/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/tools/classifier/cbsr_depth_averages/test',
                type)
            try:
                final_path = os.path.join(final_path, item.split('.')[0] + '_average_depth.png')
                file_helper.copy_file(frame_depth, final_path)
                print('    ' + frame_depth)
            except Exception as e:
                print(e)


def move_files_to_classifier_folder_all_frames(dataset_path, dataset_alias, dataset_type):
    ''''
    Required dataset structure:
    <dataset_name>:
        <type (train)>:
            <type_1 (real)>:
                <item_1>:
                    frames:
                        average_normal.jpg
                        average_depth.jpg
                <item_2>
                    frames:
                        average_normal.jpg
                        average_depth.jpg
            <type_2 (fake)>:
        <type (test)>:
            ...
    '''
    types = os.listdir(dataset_path)
    for item_type in types:
        print(item_type)

        items = os.listdir(os.path.join(dataset_path, item_type))

        threads = []
        for item in items[0:5]:
            thread = Thread(target=process_item,
                            args=(dataset_alias, dataset_path, dataset_type, item, item_type, BASE_PATH))
            threads.append(thread)
            thread.start()

            # output_path = process_item(dataset_alias, dataset_path, dataset_type, item, item_type)

        for thread in threads:  # iterates over the threads
            thread.join()  # waits until the thread has finished work


def process_item(dataset_alias, dataset_path, dataset_type, item, item_type, base_output_path):
    output_path = os.path.join(base_output_path, dataset_alias)
    path_to_look = 'depth_aligned_cropped'
    move_files(item, dataset_path, path_to_look, item_type, os.path.join(output_path, 'depth_all', dataset_type))

    path_to_look = 'illuminant_aligned_cropped'
    move_files(item, dataset_path, path_to_look, item_type, os.path.join(output_path, 'illumination_all', dataset_type))

    path_to_look = 'raw_aligned_cropped'
    move_files(item, dataset_path, path_to_look, item_type, os.path.join(output_path, 'raw_all', dataset_type))

    return output_path


cbsr_maps = [
    ['depth_aligned_cropped', 'depth'],
    ['illuminant_aligned_cropped', 'illumination'],
    ['raw_aligned_cropped', 'raw'],
    ['saliency_aligned_cropped', 'saliency']
]


def process_item_with_config(dataset_alias, dataset_path, dataset_type, item, item_type, attack_type,
                             extracted_maps=cbsr_maps,
                             base_output_path=BASE_PATH):
    threads = []

    for extracted_map in extracted_maps:
        path_to_look = extracted_map[0]
        path_to_save = os.path.join(base_output_path, dataset_alias, attack_type, extracted_map[1], dataset_type)
        thread = Thread(target=move_files, args=(item, dataset_path, path_to_look, item_type, path_to_save))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


def is_processing_done(output_path, total_frames):
    frames_done = file_helper.get_frames_from_folder(output_path)
    return len(frames_done) == total_frames


def move_files(item, dataset_path, path_to_look, type, output_path):
    frames_path = os.path.join(dataset_path, type, item, path_to_look)
    item_frames = file_helper.get_frames_from_folder(frames_path)

    threads = []
    for item_frame in item_frames:
        thread = Thread(target=move_item, args=(frames_path, item, item_frame, output_path, type))
        threads.append(thread)
        thread.start()

    for thread in threads:  # iterates over the threads
        thread.join()  # waits until the thread has finished work


from PIL import Image


def move_item(frames_depth, item, item_depth, output_path, type):
    final_name = item + '_' + item_depth
    final_path = os.path.join(output_path, type)
    item_original_path = os.path.join(frames_depth, item_depth)

    img = Image.open(item_original_path)
    width, height = img.size

    if not os.path.exists(os.path.join(final_path, final_name)) and (width == 224 and height == 224):
        file_helper.copy_file_rename(item_original_path, final_path, final_name)
        # print('copied: ', final_path, ' file name: ', final_name)
    else:
        print('bad frame, not copied: ', final_path, ' file name: ', final_name)


def move_files_to_classifier_folder_specific_frames(dataset_path, dataset_alias, dataset_type, configs):
    # TODO implement here rule for specific files
    types = os.listdir(dataset_path)

    for item_type in types:
        items = os.listdir(os.path.join(dataset_path, item_type))

        print('type:', item_type)
        for item in items:
            process_item(configs, dataset_alias, dataset_path, dataset_type, item, item_type)


def process_item(configs, dataset_alias, dataset_path, dataset_type, item, item_type):
    print('item: ', item)
    item_splitted = item.split('_')[1]
    process_item_with_config(dataset_alias, dataset_path, dataset_type, item, item_type, 'all')
    for config in configs:
        items_config = config[1]

        if item_splitted in items_config or item_type == 'real':
            process_item_with_config(dataset_alias, dataset_path, dataset_type, item, item_type, config[0])


def main():
    config_ra = [
        ['highdef', ['highdef']],
        ['mobile', ['mobile']],
        ['print', ['print']],
    ]

    move_files_to_classifier_folder_specific_frames('/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/results/ra/ra_train',
                                                    'ra', 'train', config_ra)

    move_files_to_classifier_folder_specific_frames('/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/results/ra/ra_test',
                                                    'ra', 'test', config_ra)

    # move_files_to_classifier_folder_all_frames('/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/results/ra/ra_test',
    #                                           'ra_all', 'test')

    # move_files_to_classifier_folder_all_frames('/Users/rodrigobresan/Documents/dev/github/anti_spoofing/spoopy/spoopy/static/results/ra/ra_train',
    #                                           'ra_all', 'train')


if __name__ == '__main__':
    main()
