import face_recognition
import cv2
import numpy as np


class FaceDetector:
    def __init__(self):
        self.padding = 0.2
        self.min_face_size = 30

    def extract_features(self, image):
        """Extract face and landmarks from an image"""
        try:
            if image is None:
                return None

            # Convert to RGB for face_recognition library
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get face locations
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            if not face_locations:
                print("No face detected")
                return None

            # Get facial landmarks
            face_landmarks_list = face_recognition.face_landmarks(
                rgb_image, face_locations
            )
            if not face_landmarks_list:
                print("No landmarks detected")
                return None

            # Get the largest face
            face_location = max(
                face_locations,
                key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]),
            )
            landmarks = face_landmarks_list[0]

            # Extract face coordinates with padding
            top, right, bottom, left = face_location
            height = bottom - top
            width = right - left

            # Add padding
            pad_h = int(height * self.padding)
            pad_w = int(width * self.padding)

            # Get image dimensions
            h, w = image.shape[:2]

            # Apply padding with boundary checks
            top = max(0, top - pad_h)
            bottom = min(h, bottom + pad_h)
            left = max(0, left - pad_w)
            right = min(w, right + pad_w)

            # Crop face region
            face_crop = image[top:bottom, left:right]

            # Convert landmarks to relative coordinates
            normalized_landmarks = []
            for feature in landmarks.values():
                for point in feature:
                    x, y = point
                    # Normalize coordinates relative to face crop
                    norm_x = (x - left) / width
                    norm_y = (y - top) / height
                    normalized_landmarks.extend([norm_x, norm_y])

            return face_crop, np.array(normalized_landmarks, dtype=np.float32)

        except Exception as e:
            print(f"Face detection error: {e}")
            return None
