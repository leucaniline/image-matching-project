# Image Matching Project

This project implements an image matching application that receives an image from the user and uses computer vision techniques to find lookalikes from a bank of images.

## Project Structure

```
image-matching-project
├── src
│   ├── main.py          # Entry point of the application
│   ├── image_processor   # Contains image processing logic
│   │   ├── __init__.py
│   │   ├── preprocessor.py  # Image preprocessing methods
│   │   └── matcher.py      # Image matching methods
│   ├── database           # Manages image storage
│   │   ├── __init__.py
│   │   └── image_store.py  # Storage and retrieval of images
│   └── utils             # Utility functions
│       ├── __init__.py
│       └── helpers.py     # Helper functions for various tasks
├── tests                 # Contains unit tests
│   ├── __init__.py
│   ├── test_preprocessor.py  # Tests for ImagePreprocessor
│   └── test_matcher.py      # Tests for ImageMatcher
├── data
│   └── image_bank         # Directory for image files
├── requirements.txt       # Project dependencies
├── .gitignore             # Files to ignore in Git
└── README.md              # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone (https://github.com/leucaniline/image-matching-project.git)
   ```

2. Navigate to the project directory:
   ```
   cd image-matching-project
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python src/main.py
```

Follow the prompts to upload an image and find lookalikes from the image bank.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
