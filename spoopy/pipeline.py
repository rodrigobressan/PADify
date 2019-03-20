from tools.classifier.evaluate_cross_dataset import evaluate_all_cross
from tools.data.pre_processing.pre_processing_cbsr import separate_fake_from_real, extract_frames_from_videos
from tools.map_extractor.maps_extractor import extract_maps_from_dir


def main():
    # Example for CBSR dataset
    original_path = '/codes/bresan/remote/spoopy/spoopy/data/original_videos/cbsr'
    separated_path = '/codes/bresan/remote/spoopy/spoopy/data/2_separated_videos_2/cbsr'
    frames_path = '/codes/bresan/remote/spoopy/spoopy/data/3_extracted_frames_2/cbsr'

    # separate_fake_from_real(root_dataset=original_path,
    #                         output_path=separated_path)
    #
    # extract_frames_from_videos(root_separated=separated_path,
    #                            base_output_path=frames_path)

    frames_path_all = '/codes/bresan/remote/spoopy/spoopy/data/3_extracted_frames_2'
    maps_path = '/codes/bresan/remote/spoopy/spoopy/data/4_maps_2'

    # extract_maps_from_dir(frames_base_path=frames_path_all, output_path=maps_path)

    # TODO put classify here..
    evaluate_all_cross(base_features='/codes/bresan/remote/spoopy/spoopy/data/6_features',
                       base_output='/codes/bresan/remote/spoopy/spoopy/data/8_cross')


if __name__ == '__main__':
    main()