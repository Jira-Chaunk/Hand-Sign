

# Hand Sign Classification

This project involves real-time hand sign classification using a trained RandomForest model and the MediaPipe library for hand landmark detection. The model is trained to recognize specific hand signs and can be used in various applications such as gesture-based control systems, sign language recognition, and more.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Inference](#inference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project uses a RandomForest classifier to predict hand signs based on the landmarks detected by MediaPipe. The project includes scripts for both training the model on a dataset and using the trained model for real-time hand sign recognition.

## Features

- Real-time hand landmark detection using MediaPipe.
- Hand sign classification using a pre-trained RandomForest model.
- Dynamic visualization of detected hand landmarks and predicted labels.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/handsign-classification.git
   cd handsign-classification
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the trained model:**

   Place the `model.p` file in the project directory. This file contains the trained RandomForest model.

4. **Prepare your dataset:**

   If you need to train your model, place your dataset in the `./data` directory and adjust the paths in the scripts accordingly.

## Usage

### Real-time Inference

To run the hand sign classification in real-time:

```bash
python inference_classifier.py
```

This will open a window displaying the video feed with hand landmarks drawn and the predicted hand sign shown.

### Model Training

If you want to train the model on your own dataset:

1. **Collect Data:**

   Use the `collect_imgs.py` script to collect images for different hand signs. This script will save images to the `./data` directory.

   ```bash
   python collect_imgs.py
   ```

2. **Train the Model:**

   Use the `train_classifier.py` script to train the RandomForest model on your dataset.

   ```bash
   python train_classifier.py
   ```

   The trained model will be saved in a pickle file `model.p`.

## Dataset

The dataset consists of images of different hand signs. Each class of hand signs should be stored in separate directories within the `./data` directory. The model is trained to recognize these specific classes.

### Data Collection

To collect data, use the provided script to capture images from your webcam. The script will store images in directories corresponding to different classes.

```bash
python collect_imgs.py
```

### Example Dataset Structure

```
data/
│
├── 0/  # Class A
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
│
├── 1/  # Class B
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
│
└── 2/  # Class L
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

## Model Training

The `train_classifier.py` script is used to train the RandomForest model on your dataset. The model is saved in a pickle file for later use in inference.

### Example

```bash
python train_classifier.py
```

## Inference

The `inference_classifier.py` script loads the trained model and uses your webcam to detect hand signs in real-time.

### Example

```bash
python inference_classifier.py
```

## Troubleshooting

- **Camera not opening or `NoneType` error:** Ensure the correct camera index is used in the `cv2.VideoCapture()` function. If the index is incorrect, the camera will not open, and frames will not be captured.

- **Incorrect predictions:** If the model's predictions are inaccurate, consider collecting more data or tuning the model's parameters.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to improve the project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

