<h1 align="center">🌾 PaddyNet — AI-Powered Paddy Disease Detection & Advisory Agent</h1>

<p align="center">
  A deep-learning classifier that identifies <b>10 paddy (rice) leaf conditions</b> from a photo, wrapped in an
  <b>agentic AI assistant</b> that explains the diagnosis and advises farmers on treatment — all running <b>free and offline</b>.
</p>

---

## 📌 Table of Contents

- [🔍 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🍃 Conditions Detected](#-conditions-detected)
- [📂 Dataset Overview](#-dataset-overview)
- [🧠 Model Training Pipeline](#-model-training-pipeline)
- [🏆 Results](#-results)
- [🤖 The PaddyNet Agent](#-the-paddynet-agent)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [🌐 Access from Other Devices & Deployment](#-access-from-other-devices--deployment)
- [📄 License](#-license)

---

## 🔍 Overview

**PaddyNet** is a comprehensive AI solution for rice farmers. It has two layers:

1. **The Vision Model** — an EfficientNet classifier, trained with transfer learning and 3-phase
   progressive fine-tuning, that diagnoses a leaf disease from a photo with **97.63% validation accuracy**.
2. **The Agent** — a conversational assistant ("PaddyNet") that takes the diagnosis, reasons over a curated
   agronomy knowledge base, and gives the farmer practical treatment & prevention advice — then answers
   follow-up questions in plain language.

The whole thing runs **locally and for free**: no cloud bills, no API keys, no accounts. The chat brain is a
local LLM (Ollama), with a built-in knowledge base fallback so the app works even without it.

---

## ✨ Key Features

- **🔬 10-class disease diagnosis** — from a single leaf photo, with confidence scores and full class probabilities.
- **💬 Conversational advisory agent** — ask "how do I treat it?", "will it spread?", "how do I prevent it next season?".
- **⚡ Streaming responses** — advice appears word-by-word (first text in ~3s) instead of a long blank wait.
- **🆓 100% free & offline** — local LLM via Ollama; no API keys, no billing, no data leaves your machine.
- **🛟 Graceful fallback** — if no LLM is running, a curated knowledge base still delivers structured advice.
- **🌐 Shareable** — one-click launchers for local use or access from any device on the same WiFi.

---

## 🍃 Conditions Detected

The model distinguishes **10 conditions** (8 diseases, 1 pest-damage sign, and healthy):

| # | Class | # | Class |
|---|-------|---|-------|
| 1 | Bacterial Leaf Blight | 6 | Dead Heart (stem borer) |
| 2 | Bacterial Leaf Streak | 7 | Downy Mildew |
| 3 | Bacterial Panicle Blight | 8 | Hispa |
| 4 | Blast | 9 | Normal (Healthy) |
| 5 | Brown Spot | 10 | Tungro |

---

## 📂 Dataset Overview

**Source:** [Kaggle — Paddy Disease Classification](https://www.kaggle.com/competitions/paddy-disease-classification)

| Property | Value |
|----------|-------|
| Images | ~10,000 labelled paddy leaf photos |
| Classes | 10 (class-wise folders + `train.csv` label map) |
| Input size | 300 × 300 (EfficientNet-B3) |
| Preprocessing | Resize → rescale pixels by `1/255` (matches training) |

> **Note:** the raw dataset and trained weights (`*.keras`, ~200 MB) are **not** stored in the repo due to
> GitHub's file-size limits. The model is fully reproducible by running the notebook.

---

## 🧠 Model Training Pipeline

1. **Transfer learning** — EfficientNet pre-trained on ImageNet as the backbone.
2. **3-phase progressive fine-tuning:**
   - **Phase 1** — train the classifier head with the backbone frozen.
   - **Phase 2** — unfreeze the backbone and fine-tune.
   - **Phase 3** — deep fine-tune at a low learning rate for final accuracy gains.
3. **Regularization & callbacks** — checkpoint on best val accuracy, early stopping, LR scheduling.
4. **Evaluation** — accuracy, per-class precision/recall/F1, and a confusion matrix.

The full run, plots, and metrics live in [`notebooks/paddy_disease_98 (1).ipynb`](notebooks/paddy_disease_98%20(1).ipynb).

---

## 🏆 Results

> **Best validation accuracy: `97.63%`**

| Metric | Score |
|--------|:-----:|
| **Validation Accuracy** | **97.63%** |
| Training Accuracy | 99.99% |
| Classes | 10 |
| Backbone | EfficientNet (transfer learning) |

### Accuracy per training phase

| Phase | Strategy | Validation Accuracy |
|-------|----------|:-------------------:|
| Phase 1 | Feature extraction (frozen backbone) | 63.86% |
| Phase 2 | Fine-tuning (unfrozen layers) | 97.58% |
| Phase 3 | Deep fine-tuning (low LR) | **97.63%** ✅ |

Progressive unfreezing lifted accuracy from ~64% to **~98%**.

---

## 🤖 The PaddyNet Agent

The agent turns the model from *"outputs a label"* into an assistant a farmer can talk to. It wires together
three capabilities:

```
        ┌──────────────┐     ┌────────────────────┐     ┌─────────────────────┐
Photo → │  👁️ VISION    │  →  │  📚 KNOWLEDGE       │  →  │  🧠 REASONING        │ → Advice + Chat
        │ EfficientNet │     │ Agronomy KB (10)   │     │ Local LLM (Ollama)  │
        └──────────────┘     └────────────────────┘     └─────────────────────┘
                                                          (KB-only fallback if
                                                           no LLM is running)
```

- **👁️ Vision** — `agent/model_service.py` loads the `.keras` model and predicts the disease.
- **📚 Knowledge** — `agent/knowledge_base.py` holds cause, symptoms, severity, treatment & prevention for all 10 classes.
- **🧠 Reasoning** — `agent/agent.py` grounds a local LLM on the diagnosis + KB, streams the reply, and keeps chat context.
- **🖥️ UI** — `agent/app.py` is a Streamlit web app: upload → diagnose → chat.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **ML Model** | EfficientNet-B3 · TensorFlow / Keras · transfer learning |
| **Agent / LLM** | Ollama (local, free) — `llama3.2`, with a knowledge-base fallback |
| **Web App** | Streamlit (upload + streaming chat) |
| **Data / Imaging** | NumPy · Pillow · OpenCV · Pandas |
| **Training tools** | scikit-learn · Matplotlib · Seaborn · Jupyter |

---

## 📁 Project Structure

```
paddy-disease-detection/
├── agent/                       # 🤖 PaddyNet agent (the AI layer)
│   ├── app.py                   #   Streamlit UI: upload → diagnose → chat
│   ├── agent.py                 #   Orchestrates vision + knowledge + LLM
│   ├── model_service.py         #   Loads the model, preprocesses, predicts
│   ├── knowledge_base.py        #   Agronomy facts for all 10 classes
│   ├── llm.py                   #   Local Ollama client (+ streaming)
│   ├── config.py                #   Paths, class names, model/LLM settings
│   └── README.md                #   Agent-specific docs
├── notebooks/                   # 📓 Training notebooks (EfficientNet, ResNet, MobileNet)
├── models/                      # 🧠 Trained weights + history (weights git-ignored)
├── archive/                     # 🖼️ Sample leaf images
├── data/                        # 📂 Dataset location (raw data git-ignored)
├── check_dataset.py             # 🔍 Dataset sanity check
├── model_explanation.md         # 📝 Line-by-line training write-up
├── requirements.txt             # Training dependencies
├── requirements-agent.txt       # Agent dependencies
├── run_agent.bat                # ▶️ One-click launch (local)
└── run_agent_network.bat        # 🌐 One-click launch (same-WiFi access)
```

---

## 🚀 Getting Started

### 1. Clone & set up the environment

```bash
git clone https://github.com/revankumarz/paddy-disease-detection.git
cd paddy-disease-detection

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux/Mac

pip install -r requirements.txt
pip install -r requirements-agent.txt
```

> The trained `efficientnet_final.keras` must be present in `models/`. Reproduce it by running the notebook,
> or place your own weights there.

### 2. Run the agent

```bash
streamlit run agent/app.py
```
…or on Windows just double-click **`run_agent.bat`**.

### 3. (Optional) Turn on the conversational AI — free

The app works out of the box with the knowledge base. For full conversational advice:

```bash
# Install Ollama from https://ollama.com/download, then:
ollama pull llama3.2
```
Reload the app — the sidebar shows **🧠 Local AI online**.

---

## 🌐 Access from Other Devices & Deployment

| Option | Access from |
|--------|-------------|
| **Local** | This machine |
| **Same WiFi (LAN)** | Any device on your network → run `run_agent_network.bat`, open `http://<your-ip>:8501` |
| **Internet tunnel** | Anywhere → `ngrok http 8501` gives a public URL |
| **Cloud (always-on)** | Anywhere, no host PC needed → e.g. Hugging Face Spaces + hosted model & LLM |

> The app is a server, so a machine has to run it. LAN and tunnel keep everything on your PC; a cloud deploy
> needs the model hosted externally (GitHub Release / HF Hub) and a hosted LLM in place of local Ollama.

---

## 📄 License

Released under the **MIT License**.
