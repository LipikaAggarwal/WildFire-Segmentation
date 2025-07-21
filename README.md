# 🔥 Wildfire Segmentation Using Deep Learning

This project uses a U-Net-based deep learning model to segment wildfire-affected regions from aerial imagery. It detects fire zones at the pixel level and generates binary masks that highlight the damaged areas.

Built for disaster monitoring and environmental analysis, the system helps visualize the spatial spread of wildfires. This enables timely response, risk assessment, and improved resource allocation during wildfire events.

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
