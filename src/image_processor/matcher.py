import cv2
import numpy as np
from scipy.spatial.distance import cosine


class ImageMatcher:
    def __init__(self):
        self.feature_extractor = cv2.SIFT_create()
        self.landmark_weight = 0.6
        self.sift_weight = 0.4

    def find_best_matches(self, query_data, database_images, n_matches=5):
        """Find best matches in database"""
        if query_data is None or len(database_images) == 0:
            print("No query data or empty database")
            return []

        print(f"Processing {len(database_images)} database images...")
        similarities = []

        for path, db_data in database_images:
            try:
                # Ensure both query and database data are properly structured
                if not isinstance(query_data, tuple) or not isinstance(db_data, tuple):
                    print(f"Invalid data structure for {path}")
                    continue

                # Calculate similarity between query and database image
                similarity = self.calculate_similarity(query_data, db_data)
                if similarity > 0:
                    similarities.append((similarity, path))
                    print(f"Found match: {path} ({similarity:.2f}%)")

            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        # Sort by similarity (highest first)
        similarities.sort(reverse=True)
        return similarities[:n_matches]

    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two processed images"""
        try:
            # Safely unpack tuples
            if len(features1) != 2 or len(features2) != 2:
                return 0.0

            image1, landmarks1 = features1
            image2, landmarks2 = features2

            if (
                image1 is None
                or image2 is None
                or landmarks1 is None
                or landmarks2 is None
            ):
                return 0.0

            # Calculate landmark similarity
            landmark_similarity = 1 - cosine(landmarks1.flatten(), landmarks2.flatten())

            # Extract and calculate SIFT similarity
            _, desc1 = self.feature_extractor.detectAndCompute(image1, None)
            _, desc2 = self.feature_extractor.detectAndCompute(image2, None)

            if desc1 is not None and desc2 is not None:
                desc1_mean = np.mean(desc1, axis=0)
                desc2_mean = np.mean(desc2, axis=0)
                sift_similarity = 1 - cosine(desc1_mean, desc2_mean)
            else:
                sift_similarity = 0.0

            # Combine similarities with weights
            total_similarity = (
                self.landmark_weight * landmark_similarity
                + self.sift_weight * sift_similarity
            )

            return max(0, min(100, total_similarity * 100))

        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
