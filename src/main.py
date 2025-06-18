import cv2
import os
import matplotlib.pyplot as plt
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.image_processor.preprocessor import ImagePreprocessor
from src.database.image_store import ImageStore
from src.image_processor.matcher import ImageMatcher
from tkinter import filedialog, Tk


def main():
    # Initialize components
    image_store = ImageStore(r"E:\Pictures\CSFaceMatcher")
    preprocessor = ImagePreprocessor()
    matcher = ImageMatcher()

    # Get input image from user using file dialog
    root = Tk()
    root.withdraw()  # Hide the main window
    input_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *webp"),
            ("All files", "*.*"),
        ],
    )
    if not input_path:
        raise ValueError("No file selected")

    try:
        # Read and preprocess input image
        query_image = cv2.imread(input_path)
        if query_image is None:
            raise ValueError("Could not read the input image")

        processed_query = preprocessor.process(query_image)
        if processed_query is None:
            raise ValueError("Could not process the input image")

        # Get all images from database and find best matches
        matches = matcher.find_best_matches(
            processed_query, image_store.get_all_images(), n_matches=5
        )

        # Display results
        for idx, (similarity, match_path) in enumerate(matches, 1):
            print(f"Match #{idx}: {match_path} (Similarity: {similarity:.2f}%)")

        # Display images side by side
        show_matches(input_path, matches, preprocessor)

    except Exception as e:
        print(f"Error: {str(e)}")


def show_matches(query_image_path, matches, preprocessor):
    # Load query image
    query_img = cv2.imread(query_image_path)
    if query_img is None:
        print(f"Could not read query image: {query_image_path}")
        return

    # Process query image
    processed_query = preprocessor.process(query_img)
    if processed_query is None:
        print("Could not process query image")
        return

    # Extract face image from processed tuple
    query_face, _ = processed_query
    query_rgb = cv2.cvtColor(query_face, cv2.COLOR_BGR2RGB)

    # Prepare the figure
    n_matches = len(matches)
    plt.figure(figsize=(4 * (n_matches + 1), 5))

    # Show the query image
    plt.subplot(1, n_matches + 1, 1)
    plt.imshow(query_rgb)
    plt.title("Query Image")
    plt.axis("off")

    # Show each match
    for idx, (similarity, match_path) in enumerate(matches, 2):
        match_img = cv2.imread(match_path)
        if match_img is not None:
            processed_match = preprocessor.process(match_img)
            if processed_match is not None:
                match_face, _ = processed_match
                match_rgb = cv2.cvtColor(match_face, cv2.COLOR_BGR2RGB)
                plt.subplot(1, n_matches + 1, idx)
                plt.imshow(match_rgb)
                plt.title(f"Similarity: {similarity:.2f}%")
                plt.axis("off")
            else:
                print(f"Could not process match image: {match_path}")
        else:
            print(f"Could not read match image: {match_path}")

    plt.tight_layout()
    plt.show()

    # Function to check if matches were found
    def validate_matches(matches):
        if not matches:
            print("No matches found!")
            return False
        if all(similarity < 1.0 for similarity, _ in matches):
            print("Warning: All matches have very low similarity scores")
            return False
        return True

    # Add validation check before showing matches
    if not validate_matches(matches):
        return


if __name__ == "__main__":
    main()
