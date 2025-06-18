import cv2
import numpy as np
from .facedetector import FaceDetector


class ImagePreprocessor:
    def __init__(self):
        self.target_size = (224, 224)
        self.face_detector = FaceDetector()

    def process(self, image):
        """Process image and return tuple of (processed_image, landmarks)"""
        try:
            if image is None:
                return None

            # Handle different image formats
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                elif image.shape[2] != 3:  # Unexpected number of channels
                    print(f"Unexpected image format: {image.shape}")
                    return None

            # Extract face and landmarks
            result = self.face_detector.extract_features(image)
            if result is None:
                return None

            face_crop, landmarks = result

            # Resize face crop
            resized_face = cv2.resize(face_crop, self.target_size)

            # Create final tuple with proper types
            return (resized_face.astype(np.uint8), landmarks)

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
