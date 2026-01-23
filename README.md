# Paddy Disease Detection

A deep learning project to detect diseases in paddy (rice) crops using image classification.

## Project Overview

This project aims to identify various diseases affecting paddy crops by analyzing leaf images. Early detection of diseases can help farmers take timely action and prevent crop losses.

### Diseases to Detect
- Bacterial Leaf Blight
- Brown Spot
- Leaf Smut
- Rice Blast
- Tungro
- Healthy (no disease)

## Project Structure

```
paddy-disease-detection/
├── data/
│   ├── raw/              # Original dataset
│   ├── processed/        # Preprocessed images
│   └── augmented/        # Augmented data
├── models/               # Saved trained models
├── notebooks/            # Jupyter notebooks for experiments
├── src/
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model architectures
│   ├── training/        # Training scripts
│   └── inference/       # Prediction scripts
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repository
git clone https://github.com/revankumarz/paddy-disease-detection.git
cd paddy-disease-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

Dataset can be obtained from:
- [Kaggle - Paddy Disease Classification](https://www.kaggle.com/competitions/paddy-disease-classification)

## Technologies

- Python 3.x
- TensorFlow / Keras (or PyTorch)
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn

## License

MIT License
