# 🌾 Paddy Disease Detection

Deep learning image classifier that identifies **10 paddy (rice) leaf conditions** from photographs, built with transfer learning on **EfficientNet**.

---

## 🏆 Results

> **Best validation accuracy: `97.63%`** — trained with a 3-phase progressive fine-tuning strategy on EfficientNet.

| Metric | Score |
|--------|:-----:|
| **Validation Accuracy** | **97.63%** |
| **Training Accuracy** | **99.99%** |
| Classes | 10 |
| Backbone | EfficientNet (transfer learning) |

### Accuracy per training phase

| Phase | Strategy | Validation Accuracy |
|-------|----------|:-------------------:|
| **Phase 1** | Feature extraction (frozen backbone) | 63.86% |
| **Phase 2** | Fine-tuning (unfrozen layers) | 97.58% |
| **Phase 3** | Deep fine-tuning (low LR) | **97.63%** ✅ |

The progressive unfreezing lifted accuracy from ~64% to **~98%**, with the biggest jump coming when the backbone was unfrozen and fine-tuned on the paddy dataset.

---

## 🍃 Classes Detected

The model distinguishes 10 conditions:

| # | Class | # | Class |
|---|-------|---|-------|
| 1 | Bacterial Leaf Blight | 6 | Dead Heart |
| 2 | Bacterial Leaf Streak | 7 | Downy Mildew |
| 3 | Bacterial Panicle Blight | 8 | Hispa |
| 4 | Blast | 9 | Normal (Healthy) |
| 5 | Brown Spot | 10 | Tungro |

---

## 🧠 Approach

1. **Transfer learning** — EfficientNet pre-trained on ImageNet as the backbone.
2. **3-phase progressive fine-tuning:**
   - Phase 1: train the classifier head with the backbone frozen.
   - Phase 2: unfreeze the backbone and fine-tune.
   - Phase 3: deep fine-tune at a low learning rate for the final accuracy gains.
3. **Regularization & callbacks** — checkpointing on best validation accuracy, early stopping, and learning-rate scheduling.
4. **Evaluation** — accuracy, per-class precision/recall/F1, and a confusion matrix.

The full training run, plots, and metrics live in
[`notebooks/paddy_disease_98 (1).ipynb`](notebooks/paddy_disease_98%20(1).ipynb).

---

## 📁 Project Structure

```
paddy-disease-detection/
├── notebooks/
│   ├── paddy_disease_98 (1).ipynb          # ⭐ Main run — 97.63% val accuracy (EfficientNet)
│   ├── paddy_disease_detection_fixed.ipynb # Baseline / experiments
│   ├── paddy_disease_mobilenet.ipynb       # MobileNet comparison
│   └── paddy_disease_resnet_gpu.ipynb      # ResNet comparison
├── archive/                                # Sample images by class
├── check_dataset.py                        # Dataset sanity check
├── model_explanation.md                    # Training process write-up
├── paddy_presentation_updated.odp          # Project presentation
├── requirements.txt
└── README.md
```

> **Note:** Trained model weights (`.keras`, ~200 MB each) and the raw dataset are **not** stored in this repo due to GitHub's file-size limits. The models are fully reproducible by running the notebook. Training-history files are included under `models/`.

---

## 🚀 Reproduce

```bash
# 1. Clone
git clone https://github.com/revankumarz/paddy-disease-detection.git
cd paddy-disease-detection

# 2. Environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt

# 3. Download the dataset from Kaggle into data/raw/
#    https://www.kaggle.com/competitions/paddy-disease-classification

# 4. Run the notebook
jupyter notebook "notebooks/paddy_disease_98 (1).ipynb"
```

---

## 🛠️ Tech Stack

Python · TensorFlow / Keras · EfficientNet · NumPy · Pandas · scikit-learn · Matplotlib · Seaborn · Jupyter

---

## 📊 Dataset

**Source:** [Kaggle — Paddy Disease Classification](https://www.kaggle.com/competitions/paddy-disease-classification)
~10,000 labelled paddy leaf images across 10 classes, resized for EfficientNet input.

---

## License

MIT
