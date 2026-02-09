# pneumonia_detection_project
Repository for my pneumonia detection project
# 🫁 Pneumonia Detection from Chest X-Rays using Transfer Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![VGG16](https://img.shields.io/badge/Model-VGG16-green?style=flat-square)

## 📌 Project Overview
Pneumonia is a life-threatening condition that requires rapid and accurate diagnosis. It is responsible for 14% of deaths of children under the age of 5 and it is one of the major causes of death in both infants and elderly people. It is more common in poorer countries with high overcrowding and levels of pollution.
In this project, machine learning models were made to automatically detect pneumonia from chest X-ray images using the **VGG16** and **DenseNet121**  **Convolutional Neural Networks (CNN)s**. These two models were then compared to see which performed. Finetuning of the final layers of each model architecture was also undertaken.
A successful model to automatically detect pneumonia could help as a potential decision-support tool for medical professionals in detecting pneumonia faster. This would be especially helpful in poorer countries where pneuomia is more prevalent whilst the number of healthcare professionals is more limited.

## 📂 Dataset
The dataset was obtained from [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
- **Total Images:** 5,863 JPEG images
- **Categories:** Normal vs. Pneumonia (Bacterial & Viral)
- **Data Split:** Train / Test / Validation

## ⚙️ Methodology
1.  **Data Preprocessing:**
    - Resized to 224x224 (standard input for VGG16).
    - Normalization (pixel values rescaled to 0-1).
    - **Data Augmentation** (rotation, zoom, flips) applied to training data to prevent overfitting. Test data was not augmented.
2.  **Model Architecture:**
    - **Base:** For both models, pretrained weights from training on the ImageNet database were used. The layers of both models were frozen. 
    - **Head:** A custom head was added to both models to adapt the pre-trained model feature extraction capabilities to the new pneumonia dataset. The custom heads included a GlobalAveragePooling2D layer, a 128 node dense layer, a dropout layer (0.5) to prevent overfitting and a dense sigmoid activation final output layer.
      
4.  **Training Strategy:**
    - Used **Class Weights** to handle the imbalance between 'Pneumonia' (majority) and 'Normal' (minority) classes.
    - Optimizer: Adam (LR=0.0001).
    - Loss Function: Binary Crossentropy.
    - Trained base models for 10 epochs and finetuned models for 5 epochs.
5. **Finetuning:**
    - The last block in VGG16 (Block 5) was finetuned. The last 50 layers were finetuned in DenseNet121.
    - A low learning rate of 1 * 10<sup>-6</sup> was used for both models. VGG16 was also finetuned with a learning rate of 1 * 10<sup>-5</sup>.

## 📊 Findings
<div align="center">

| Model Architecture | Status | Accuracy | Recall (Sensitivity) |
| :--- | :--- | :--- | :--- |
| **VGG16** | Base (Frozen) | *84 %* | *83 %* |
| **VGG16** | Fine-Tuned (10<sup>-5</sup>) | *85 %* | *85 %* |
| **VGG16** | Fine-Tuned (10<sup>-6</sup>) | *89 %* | *91 %* |
| **DenseNet121** | Base (Frozen) | *89 %* | *92 %* |
| **DenseNet121** | Fine-Tuned | *90 %* | *92 %* |

<i>Table showing the accuracy and recall values for each model model created in this project. Accuracy is the percentage of total correct predictions and recall is the percentage of actual pneumonia cases. successfully detected</i>
<div align="left">

- Overall, models based on the DenseNet121 models have higher accuracy and recall values then the models based on the VGG16 architecture.
- The Finetuned DenseNet121 model had the best scores overall.
- The more gently finetuned VGG16 model had better results than the more extremely finetuned model. This shows the importance of not over-finetuning. Finetuning had a greater effect in improving the VGG16 rather than the DenseNet121 model.

<div align="center">
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/c604c00c-2af5-4429-8c7c-bd7dd4197c5d" />
   <div align="center">
<i>Table showing the recall and accuracy values for the VGG16 based models.</i>
<div align="left">

# 🧠 ![Explainable AI (Grad-CAM)]()
To improve the explainability of the models and to make sure that the models were looking at the right features, **Grad-CAM (Gradient-weighted Class Activation Mapping)** was used on all the models.
The same X-ray image was used for each model Grad-CAM for consistency. Grad-CAM creates a heatmap with the areas of the image were the model focused on in red, and where it did not in blue.

<div align="center">
<img width="950" height="315" alt="image" src="https://github.com/user-attachments/assets/ff096e7c-668d-412e-9661-12209da66a38" />
<div align="center">
<i>Figure showing the Grad-CAM heatmap for the VGG16 model without any finetuning created in this project.</i>
<div align="left">
<br>

<div align="center">
<img width="950" height="315" alt="image" src="https://github.com/user-attachments/assets/4ad4d0f2-c047-43e5-ada4-dec68130b10c" />
<div align="center">
<i>Figure showing the Grad-CAM heatmap for the VGG16 model with gentle (10<sup>-5</sup>) finetuning created in this project.</i>
<div align="left">
<br>

<div align="center">
<img width="950" height="315" alt="image" src="https://github.com/user-attachments/assets/bb579f0e-b545-4fad-b80f-a90d84ca5082" />
<div align="center">
<i>Figure showing the Grad-CAM heatmap for the DenseNet121 model without any finetuning created in this project.</i>
<div align="left">
<br>

<div align="center">
<img width="950" height="315" alt="image" src="https://github.com/user-attachments/assets/8fa815de-95f8-4e3b-9343-22f6c53fd62b" />
<div align="center">
<i>Figure showing the Grad-CAM heatmap for the DenseNet121 model with gentle (10<sup>-5</sup>) finetuning created in this project.</i>
<div align="left">
<br>


The heatmap below shows the model focusing heavily on the lung opacity (cloudiness), confirming it is learning clinically relevant features.


    
### ![Confusion Matrix]()
The model demonstrates a strong ability to distinguish between infected and healthy lungs, with a priority on minimizing False Negatives (missed cases).

<div align="center">
<img width="528" height="547" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/4432f345-e2eb-4e8e-9116-6a000e20ab1b" />
    <div align="center">
<i>Figure showing the confusion matrix for the model created in this project.</i>
<div align="left">




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
