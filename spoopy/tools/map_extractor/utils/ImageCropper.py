class ImageCropper:
    def __init__(self, face_cropper, face_coordinates):
        self.face_cropper = face_cropper
        self.face_coordinates = face_coordinates

    def crop(self, img, output_path):
        self.face_cropper.crop_image_into_face(img, self.face_coordinates, output_path)
