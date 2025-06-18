import cv2
import numpy as np
from pathlib import Path
from src.image_processor.preprocessor import ImagePreprocessor


class ImageStore:
    def __init__(self, image_directory):
        self.image_directory = Path(image_directory)
        self.supported_formats = (".png", ".jpg", ".jpeg")
        self.preprocessor = ImagePreprocessor()

    def get_all_images(self):
        """Returns list of tuples (image_path, (processed_image, landmarks))"""
        images = []
        print("Starting database image processing...")

        for img_path in self.image_directory.rglob("*"):
            if img_path.suffix.lower() in self.supported_formats:
                try:
                    # Read image with IMREAD_UNCHANGED for PNG support
                    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        print(f"Failed to load: {img_path}")
                        continue

                    # Handle RGBA images
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                    # Process image and get face features
                    processed_data = self.preprocessor.process(img)
                    if (
                        processed_data is not None
                        and isinstance(processed_data, tuple)
                        and len(processed_data) == 2
                    ):
                        # Store as (path, (processed_image, landmarks))
                        images.append((str(img_path), processed_data))
                        print(f"Successfully processed: {img_path}")
                    else:
                        print(f"Invalid processing result for: {img_path}")

                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue

        print(
            f"Database loading complete. Processed {len(images)} images successfully."
        )
        return images
