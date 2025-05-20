# Pneumonia Detection using Deep Learning

This project implements a deep learning model to detect pneumonia from chest X-ray images using EfficientNetB3. The model is trained to classify chest X-ray images into two categories: Normal and Pneumonia.

## Features

- Deep learning model based on EfficientNetB3 architecture
- Data augmentation for improved model performance
- Comprehensive visualization of results
- Support for both English and Arabic text
- Model evaluation metrics and visualizations
- Layer activation visualization
- Test predictions on new images

## Requirements

- Python 3.8 or higher
- Required packages (see requirements.txt):
  - numpy>=1.19.2
  - pandas>=1.2.0
  - matplotlib>=3.3.2
  - seaborn>=0.11.0
  - tensorflow>=2.4.0
  - scikit-learn>=0.24.0
  - opencv-python>=4.4.0
  - kaggle>=1.5.12
  - tqdm>=4.50.0
  - arabic-reshaper>=2.1.3
  - python-bidi>=0.4.2

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mahmoudnasraly/pneumonia-detection.git
cd pneumonia-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle. You can download it from:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Expected folder structure:
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Usage

### Running the Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `pneumonia_detection.ipynb`

3. Run the cells in order (Shift+Enter)

### Running the Python Script

```bash
python pneumonia_detection.py
```

## Project Structure

```
pneumonia-detection/
├── pneumonia_detection.ipynb    # Jupyter notebook version
├── pneumonia_detection.py       # Python script version
├── arabic_text_setup.py        # Arabic text handling utilities
├── requirements.txt            # Project dependencies
├── README.md                   # This file
└── chest_xray/                 # Dataset directory
```

## Model Architecture

The model uses EfficientNetB3 as the base architecture with the following modifications:
- Global Average Pooling layer
- Dense layers (512 and 256 units) with ReLU activation
- Dropout layers (0.5 and 0.3) for regularization
- Final sigmoid activation for binary classification

## Training Process

1. Initial training with frozen base model
2. Fine-tuning with unfrozen layers
3. Data augmentation for improved generalization
4. Early stopping and learning rate reduction

## Results

The model provides:
- Classification accuracy
- Confusion matrix
- ROC curve and AUC score
- Training/validation curves
- Sample predictions
- Layer activation visualizations

## Arabic Text Support

The project includes support for Arabic text in visualizations and outputs. This is handled by:
- `arabic-reshaper` for proper character reshaping
- `python-bidi` for text direction handling
- System Arabic fonts for proper rendering

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: Chest X-Ray Images (Pneumonia) by Paul Mooney
- EfficientNet paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
- Arabic text handling libraries: arabic-reshaper and python-bidi

## Contact

For questions and suggestions, please open an issue in the repository. 
