# Face Recognition using ArcFace Embeddings + Deep Neural Network

A complete face recognition pipeline that uses **ArcFace** (via DeepFace) to extract 512-dimensional facial embeddings from the **LFW (Labeled Faces in the Wild)** dataset, then trains a custom **Multi-Layer Perceptron (MLP)** classifier achieving ~98% accuracy across 42 identity classes.

---

## Overview

This project implements a two-stage face recognition system:

1. **Feature Extraction** — ArcFace model extracts a rich 512-dim embedding from each face image, capturing fine-grained identity features
2. **Classification** — A custom MLP trained on L2-normalized embeddings recognizes the correct identity among 42 people

The approach avoids training a CNN from scratch by leveraging a pre-trained state-of-the-art face recognition model (ArcFace) as a feature extractor, making it computationally efficient while achieving high accuracy.

---

## Dataset

| Property | Value |
|---|---|
| Dataset | Labeled Faces in the Wild (LFW) |
| Source | `sklearn.datasets.fetch_lfw_people` |
| Min faces per person | 25 |
| Total Images | 2,588 |
| Total Classes (Identities) | 42 |
| Image Shape | 125 × 94 (grayscale) |
| Train / Test Split | 80% / 20% (2,070 / 518) |

> The dataset is automatically downloaded on first run via scikit-learn — no manual download needed.

---

## Pipeline

```
Raw LFW Images (grayscale, 125×94)
        │
        ▼
Grayscale → BGR Conversion (OpenCV)
        │
        ▼
ArcFace Embedding Extraction via DeepFace
  → 512-dimensional embedding per image
  → enforce_detection=False (handles low-res images)
  → Filter near-zero embeddings (norm < 1e-3)
        │
        ▼
L2 Normalization (sklearn.preprocessing.normalize)
        │
        ▼
Train / Test Split (80/20, stratified)
        │
        ▼
MLP Classifier Training (50 epochs, Adam, ReduceLROnPlateau)
        │
        ▼
Evaluation → Accuracy Score + Classification Report
        │
        ▼
Predictions on Sample Images with Visualization
```

---

## Model Architecture

The MLP classifier is built on top of frozen ArcFace embeddings:

```
Input: 512-dimensional ArcFace embedding (L2-normalized)
│
├── Dense(512) → BatchNorm → ReLU → Dropout
├── Dense(256) → BatchNorm → ReLU → Dropout
└── Dense(42)  → Softmax
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Dense (512, ReLU) | (None, 512) | 262,656 |
| BatchNormalization | (None, 512) | 2,048 |
| Dropout | (None, 512) | 0 |
| Dense (256, ReLU) | (None, 256) | 131,328 |
| BatchNormalization | (None, 256) | 1,024 |
| Dropout | (None, 256) | 0 |
| Dense (42, Softmax) | (None, 42) | 10,794 |

**Total Parameters:** 407,850 (~1.56 MB)

**Training Config:**
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Callbacks: ReduceLROnPlateau, EarlyStopping
- Epochs: 50

---

## 📈 Results

| Metric | Value |
|---|---|
| Training Accuracy | ~98% |
| Validation Accuracy | ~98% |
| Valid Embeddings Extracted | 2,588 / 2,588 |
| Failed Embeddings | 0 |

---

## Tech Stack

| Library | Purpose |
|---|---|
| `TensorFlow / Keras` | MLP model building & training |
| `DeepFace` | ArcFace embedding extraction |
| `OpenCV (cv2)` | Image preprocessing (grayscale → BGR) |
| `scikit-learn` | Dataset loading, train/test split, label encoding, normalization |
| `NumPy` | Array & numerical operations |
| `Matplotlib` | Training curves & prediction visualization |

---

##  Installation & Usage

**Step 1 — Clone the repository**
```bash
git clone https://github.com/your-username/face-recognition-arcface.git
cd face-recognition-arcface
```

**Step 2 — Run the notebook**
```bash
jupyter notebook HumanFaceRecognition.ipynb
```
---

## 📁 Project Structure

```
face-recognition-arcface/
│
├── dnnass6.ipynb        # Main Jupyter notebook (complete pipeline)
└── README.md            # Project documentation
```
---

## 👤 Author

**Vaishnavi Sapkal**  
