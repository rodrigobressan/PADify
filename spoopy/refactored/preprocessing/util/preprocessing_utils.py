from tools.file_utils import file_helper


def get_not_processed_frames(all_frames_path, output_path):
    all_images = file_helper.get_frames_from_folder(all_frames_path)
    processed = file_helper.get_frames_from_folder(output_path)

    missing_frames = [frame for frame in all_images if frame not in processed]
    return missing_frames
