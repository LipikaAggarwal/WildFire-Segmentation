# 🔥 Wildfire Segmentation Using Deep Learning

This project focuses on segmenting wildfire-affected regions from aerial imagery using a deep learning model. It identifies fire zones at the pixel level, making it easier to visualize the spatial extent of wildfire damage.

The model is based on a U-Net architecture and is trained to generate binary masks that highlight fire-affected areas in the input imagery.

---

## 🧠 What the Project Does

Given an aerial image as input, the model outputs a binary segmentation mask where:

- **White pixels** indicate fire-affected regions  
- **Black pixels** indicate unaffected regions

This approach supports visual analysis, disaster monitoring, and research into wildfire impact detection using aerial or satellite-style data.

---

## 🖼️ Sample Predictions

| Input Image | Predicted Mask |
|-------------|----------------|
| ![](assets/sample1_input.png) | ![](assets/sample1_output.png) |
| ![](assets/sample2_input.png) | ![](assets/sample2_output.png) |

---

## 📦 Dataset

- **Name**: Fire Segmentation Dataset  
- **Source**: [Kaggle – Fire Segmentation Dataset](https://www.kaggle.com/datasets/killa92/fire-segmentation-dataset?utm_source=chatgpt.com)  
- **Type**: Aerial images with corresponding binary fire masks  
- **Usage**: Used for **supervised learning** in wildfire segmentation tasks

---
