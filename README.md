# Paddy Disease Detection

A deep learning project to detect diseases in paddy (rice) crops using image classification.

## Project Overview

This project aims to identify various diseases affecting paddy crops by analyzing leaf images. Early detection of diseases can help farmers take timely action and prevent crop losses.

### Diseases to Detect
| Class | Description |
|-------|-------------|
| Bacterial Leaf Blight | Bacterial infection causing yellow-orange lesions |
| Brown Spot | Fungal disease with brown oval spots |
| Leaf Smut | Fungal infection causing black sooty masses |
| Rice Blast | Fungal disease causing diamond-shaped lesions |
| Tungro | Viral disease causing yellow-orange discoloration |
| Healthy | No disease present |

---

## Complete Project Flow

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PADDY DISEASE DETECTION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────────┐     │
│   │  DATA    │───►│ PREPROCESSING│───►│ TRAINING │───►│  INFERENCE   │     │
│   │ (Images) │    │  & AUGMENT   │    │ PIPELINE │    │  (Predict)   │     │
│   └──────────┘    └──────────────┘    └──────────┘    └──────────────┘     │
│        │                 │                  │                │              │
│        ▼                 ▼                  ▼                ▼              │
│   data/raw/         data/processed/    models/         Predictions         │
│                     data/augmented/                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Data Flow

```
                              ┌─────────────────────┐
                              │   Raw Dataset       │
                              │   (Kaggle)          │
                              └──────────┬──────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │   data/raw/         │
                              │   - train_images/   │
                              │   - test_images/    │
                              │   - train.csv       │
                              └──────────┬──────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
         ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
         │  Load Images     │ │  Resize Images   │ │  Normalize       │
         │  (OpenCV/PIL)    │ │  (224x224)       │ │  (0-1 scale)     │
         └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
                  │                    │                    │
                  └────────────────────┼────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────────┐
                              │   Data Augmentation │
                              │   - Rotation        │
                              │   - Flip            │
                              │   - Zoom            │
                              │   - Brightness      │
                              └──────────┬──────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
                    ▼                                         ▼
         ┌──────────────────┐                      ┌──────────────────┐
         │ data/processed/  │                      │ data/augmented/  │
         │ (Clean images)   │                      │ (Extra samples)  │
         └────────┬─────────┘                      └────────┬─────────┘
                  │                                         │
                  └──────────────────┬──────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │   Train/Val Split   │
                          │   (80/20)           │
                          └──────────┬──────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
         ┌──────────────────┐              ┌──────────────────┐
         │  Training Set    │              │  Validation Set  │
         └────────┬─────────┘              └────────┬─────────┘
                  │                                 │
                  └────────────────┬────────────────┘
                                   │
                                   ▼
                          ┌─────────────────────┐
                          │   Model Training    │
                          │   (CNN/Transfer)    │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │   models/           │
                          │   - best_model.h5   │
                          │   - final_model.h5  │
                          └──────────┬──────────┘
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │   Inference         │
                          │   (Predictions)     │
                          └─────────────────────┘
```

---

## Project Structure

```
paddy-disease-detection/
├── data/
│   ├── raw/                    # Original dataset from Kaggle
│   │   ├── train_images/       # Training images organized by class
│   │   ├── test_images/        # Test images for submission
│   │   └── train.csv           # Labels and metadata
│   ├── processed/              # Preprocessed images (resized, normalized)
│   └── augmented/              # Augmented training data
│
├── models/                     # Saved trained models
│   ├── checkpoints/            # Training checkpoints
│   ├── best_model.h5           # Best model based on validation accuracy
│   └── final_model.h5          # Final trained model
│
├── notebooks/                  # Jupyter notebooks for experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── data/                   # Data loading and preprocessing modules
│   │   ├── __init__.py
│   │   ├── dataloader.py       # Custom data loaders
│   │   ├── preprocessing.py    # Image preprocessing functions
│   │   └── augmentation.py     # Data augmentation pipelines
│   │
│   ├── models/                 # Model architecture definitions
│   │   ├── __init__.py
│   │   ├── cnn_model.py        # Custom CNN architecture
│   │   └── transfer_learning.py # Pre-trained model implementations
│   │
│   ├── training/               # Training scripts and utilities
│   │   ├── __init__.py
│   │   ├── train.py            # Main training script
│   │   ├── callbacks.py        # Custom callbacks (early stopping, etc.)
│   │   └── losses.py           # Custom loss functions
│   │
│   └── inference/              # Prediction scripts
│       ├── __init__.py
│       ├── predict.py          # Single image prediction
│       └── batch_predict.py    # Batch prediction for test set
│
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## Module Descriptions

### 1. Data Module (`src/data/`)

| File | Purpose |
|------|---------|
| `dataloader.py` | Load images and labels, create train/validation generators |
| `preprocessing.py` | Resize, normalize, and clean images |
| `augmentation.py` | Apply transformations to increase dataset diversity |

**Key Functions:**
- Load images from directory structure
- Apply image preprocessing (resize to 224x224, normalize to 0-1)
- Create data augmentation pipelines
- Generate train/validation splits

### 2. Models Module (`src/models/`)

| File | Purpose |
|------|---------|
| `cnn_model.py` | Custom CNN architecture from scratch |
| `transfer_learning.py` | Pre-trained models (ResNet, EfficientNet, VGG) |

**Supported Architectures:**
- Custom CNN with Conv2D, MaxPooling, Dense layers
- ResNet50 (Transfer Learning)
- EfficientNetB0-B7 (Transfer Learning)
- VGG16/VGG19 (Transfer Learning)

### 3. Training Module (`src/training/`)

| File | Purpose |
|------|---------|
| `train.py` | Main training loop and configuration |
| `callbacks.py` | Model checkpointing, early stopping, learning rate scheduling |
| `losses.py` | Custom loss functions (focal loss for class imbalance) |

**Training Features:**
- Configurable epochs, batch size, learning rate
- Early stopping to prevent overfitting
- Model checkpointing to save best weights
- Learning rate scheduling for optimal convergence

### 4. Inference Module (`src/inference/`)

| File | Purpose |
|------|---------|
| `predict.py` | Single image prediction with confidence scores |
| `batch_predict.py` | Batch predictions for test dataset |

**Inference Features:**
- Load trained model and predict disease class
- Return confidence scores for all classes
- Generate submission CSV for Kaggle

---

## Step-by-Step Workflow

### Phase 1: Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/revankumarz/paddy-disease-detection.git
cd paddy-disease-detection

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Phase 2: Data Preparation

```bash
# 1. Download dataset from Kaggle
# Visit: https://www.kaggle.com/competitions/paddy-disease-classification
# Download and extract to data/raw/

# 2. Verify data structure
data/raw/
├── train_images/
│   ├── bacterial_leaf_blight/
│   ├── brown_spot/
│   ├── leaf_smut/
│   ├── rice_blast/
│   ├── tungro/
│   └── healthy/
├── test_images/
└── train.csv
```

### Phase 3: Data Preprocessing

```python
# Run preprocessing script
from src.data.preprocessing import preprocess_dataset

# This will:
# - Resize all images to 224x224
# - Normalize pixel values to 0-1
# - Save processed images to data/processed/
preprocess_dataset(
    input_dir='data/raw/train_images',
    output_dir='data/processed',
    target_size=(224, 224)
)
```

### Phase 4: Data Augmentation

```python
# Apply augmentation to increase training data
from src.data.augmentation import augment_dataset

# Augmentation techniques:
# - Random rotation (0-30 degrees)
# - Horizontal/Vertical flip
# - Random zoom (0.8-1.2x)
# - Brightness adjustment
augment_dataset(
    input_dir='data/processed',
    output_dir='data/augmented',
    augmentation_factor=3  # Create 3x more samples
)
```

### Phase 5: Model Training

```python
# Train the model
from src.training.train import train_model

# Configure training parameters
config = {
    'model_type': 'efficientnet',  # or 'resnet', 'custom_cnn'
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'train_dir': 'data/processed',
    'val_split': 0.2
}

# Start training
history = train_model(config)
# Model saved to models/best_model.h5
```

### Phase 6: Model Evaluation

```python
# Evaluate model performance
from src.training.train import evaluate_model

# Get metrics
metrics = evaluate_model(
    model_path='models/best_model.h5',
    test_dir='data/processed'
)

# Output:
# - Accuracy
# - Precision, Recall, F1-Score
# - Confusion Matrix
# - Classification Report
```

### Phase 7: Inference

```python
# Single image prediction
from src.inference.predict import predict_disease

result = predict_disease(
    image_path='path/to/leaf/image.jpg',
    model_path='models/best_model.h5'
)

print(f"Disease: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction for Kaggle submission
from src.inference.batch_predict import generate_submission

generate_submission(
    test_dir='data/raw/test_images',
    model_path='models/best_model.h5',
    output_path='submission.csv'
)
```

---

## Technologies

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Language** | Python | 3.8+ | Core programming language |
| **Deep Learning** | TensorFlow | >= 2.10.0 | Neural network framework |
| **Deep Learning** | Keras | (included) | High-level API for TensorFlow |
| **Image Processing** | OpenCV | >= 4.5.0 | Image loading and manipulation |
| **Image Processing** | Pillow | >= 8.0.0 | Image preprocessing |
| **Data Science** | NumPy | >= 1.21.0 | Numerical computations |
| **Data Science** | Pandas | >= 1.3.0 | Data manipulation |
| **Machine Learning** | Scikit-learn | >= 1.0.0 | Metrics and utilities |
| **Visualization** | Matplotlib | >= 3.4.0 | Plotting and charts |
| **Visualization** | Seaborn | >= 0.11.0 | Statistical visualizations |
| **Development** | Jupyter | >= 1.0.0 | Interactive notebooks |
| **Utilities** | TQDM | >= 4.62.0 | Progress bars |

---

## Dataset

**Source:** [Kaggle - Paddy Disease Classification](https://www.kaggle.com/competitions/paddy-disease-classification)

**Dataset Statistics:**
- Total training images: ~10,000
- Number of classes: 6
- Image format: JPEG
- Original resolution: Various (resized to 224x224)

**Class Distribution:**
| Class | Approximate Count |
|-------|-------------------|
| Bacterial Leaf Blight | ~1,500 |
| Brown Spot | ~1,500 |
| Leaf Smut | ~1,500 |
| Rice Blast | ~1,500 |
| Tungro | ~1,500 |
| Healthy | ~2,500 |

---

## Expected Results

| Metric | Target |
|--------|--------|
| Training Accuracy | > 95% |
| Validation Accuracy | > 90% |
| Test Accuracy | > 85% |

---

## Future Enhancements

- [ ] Add REST API for model deployment
- [ ] Mobile app integration
- [ ] Real-time disease detection from camera feed
- [ ] Support for additional crop diseases
- [ ] Model optimization for edge devices

---

## License

MIT License

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
