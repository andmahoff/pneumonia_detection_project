# pneumonia_detection_project
Repository for my pneumonia detection project
# 🫁 Pneumonia Detection from Chest X-Rays using Transfer Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![VGG16](https://img.shields.io/badge/Model-VGG16-green?style=flat-square)

## 📌 Project Overview
Pneumonia is a life-threatening condition that requires rapid and accurate diagnosis. Radiologists face high workloads, increasing the risk of fatigue-related errors. This project implements a **Convolutional Neural Network (CNN)** based on the **VGG16** architecture to automatically detect pneumonia from chest X-ray images, serving as a potential decision-support tool for medical professionals.

## 📂 Dataset
The dataset was obtained from [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
- **Total Images:** 5,863 JPEG images
- **Categories:** Normal vs. Pneumonia (Bacterial & Viral)
- **Data Split:** Train / Test / Validation

## ⚙️ Methodology
1.  **Data Preprocessing:**
    - Resizing to 224x224 (standard input for VGG16).
    - Normalization (pixel values rescaled to 0-1).
    - **Data Augmentation** (rotation, zoom, flips) applied to training data to prevent overfitting.
2.  **Model Architecture:**
    - **Base:** Pre-trained VGG16 (weights from ImageNet) with layers frozen.
    - **Head:** Custom fully connected layers (GlobalAveragePooling -> Dense(128) -> Dropout(0.5) -> Sigmoid).
3.  **Training Strategy:**
    - Used **Class Weights** to handle the imbalance between 'Pneumonia' (majority) and 'Normal' (minority) classes.
    - Optimizer: Adam (LR=0.0001).
    - Loss Function: Binary Crossentropy.

## 📊 Key Results
| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **85.3%** |
| **Recall (Sensitivity)** | **87.0%** |
| **Precision** | **89.0%** |

### Confusion Matrix
The model demonstrates a strong ability to distinguish between infected and healthy lungs, with a priority on minimizing False Negatives (missed cases).

![Confusion Matrix]
<div align="center">
<img width="528" height="547" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/4432f345-e2eb-4e8e-9116-6a000e20ab1b" />
    <div align="center">
<i>Figure showing the confusion matrix for the model created in this project.</i>
<div align="left">

## 🧠 Explainable AI (Grad-CAM)
To ensure the model isn't "cheating" by looking at artifacts (like text labels or bone structures), I implemented **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

The heatmap below shows the model focusing heavily on the lung opacity (cloudiness), confirming it is learning clinically relevant features.

![Grad-CAM Visualization](URL_TO_YOUR_GRADCAM_IMAGE)
<img width="950" height="315" alt="CMAP graph" src="https://github.com/user-attachments/assets/3ecfbc26-93a6-4fc6-bb1d-e8a28049b358" />


## 💡 Challenges Overcome
**The "Shuffling" Bug:**
During the initial evaluation, the test accuracy dropped unexpectedly to ~54% (near random guessing). Upon debugging, I discovered that the standard Keras `flow_from_directory` method shuffles data by default. This caused a mismatch between the model's predictions and the ordered list of ground truth labels.
* **Fix:** Re-initialized the test generator with `shuffle=False` to align predictions with true labels, restoring accuracy to 85.3%.

## 🚀 How to Run
1.  Clone the repository.
2.  Install dependencies: `pip install tensorflow numpy matplotlib seaborn opencv-python`.
3.  Download the dataset from Kaggle and place it in the root directory.
4.  Run the Jupyter Notebook `Pneumonia_Detection.ipynb`.

---
*Created by Andrzej Machowski*
