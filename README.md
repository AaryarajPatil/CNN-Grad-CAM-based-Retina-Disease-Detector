# CNN & Grad-CAM based Retina Disease Detector

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An explainable deep learning pipeline for multi-class retinal disease detection from fundus images using **MobileNetV2** transfer learning and **Grad-CAM (Gradient-weighted Class Activation Mapping)** for visual interpretability.

Developed as an Engineering Design and Innovation (EDI) project.

---

## Overview

Retinal conditions such as Diabetic Retinopathy and Glaucoma are leading causes of visual impairment and blindness globally. Early diagnosis is critical to preventing irreversible vision loss. This project delivers an automated, high-accuracy classification pipeline paired with visual explanations (Grad-CAM heatmaps) to assist healthcare professionals in understanding model decisions.

### Target Classes
1. **Diabetic Retinopathy**
2. **Glaucoma**
3. **Normal (Healthy Retina)**

---

## Visual Results & Explainability

### Grad-CAM Interpretability
To ensure transparency and clinical trust, the model utilizes Grad-CAM on the final convolutional layer of MobileNetV2 to highlight retinal regions (optic disc, blood vessels, microaneurysms) driving the prediction:

<p align="center">
  <img src="results/Combined%20Grad-CAM%20image.png" alt="Combined Grad-CAM Visualization" width="850"/>
</p>

### Training & Validation Performance
The model was trained for 10 epochs using transfer learning, achieving **~87% validation accuracy** with stable convergence and no significant overfitting:

<p align="center">
  <img src="results/Accuracy%20graph%20image.png" alt="Training and Validation Accuracy Graph" width="600"/>
</p>

---

## Model Architecture & Pipeline

- **Backbone Architecture:** MobileNetV2 (pretrained on ImageNet), frozen feature extractor
- **Input Dimensions:** `224 x 224 x 3`
- **Classification Head:**
  - `GlobalAveragePooling2D`
  - `Dropout(rate=0.3)`
  - `Dense(3, activation='softmax')`
- **Optimization:** Adam optimizer (`lr=1e-4`)
- **Loss Function:** Categorical Crossentropy
- **Data Augmentation & Preprocessing:** Rescaling (`1/255`), 80% train / 20% validation split

---

## Dataset

The dataset was curated by combining two publicly available retinal fundus photography datasets:
- **Filtering & Quality Control:** Removal of low-resolution, blurred, or corrupt images.
- **Deduplication:** Filtering out duplicate samples across subsets.
- **Organization:** Class-balanced folders for training and validation.

> *Note: Due to licensing terms and patient privacy constraints, raw dataset images are not hosted directly in this repository. Refer to [`data/README.md`](data/README.md) for further information.*

---

## Repository Structure

```text
├── data/
│   └── README.md              # Dataset notes and licensing details
├── notebooks/
│   ├── CNN and Grad-CAM based Multi Retina Disease Detector.ipynb  # Main pipeline notebook
│   └── README.md              # Notebook walkthrough and execution instructions
├── results/
│   ├── Accuracy graph image.png    # Training vs validation accuracy plot
│   ├── Combined Grad-CAM image.png # Original vs Heatmap vs Grad-CAM overlay
│   └── README.md              # Summary of outputs and visualizations
├── .gitignore                 # Standard python and deep learning ignores
├── LICENSE                    # MIT License
├── README.md                  # Project overview and documentation
└── requirements.txt           # Python dependencies
```

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- GPU environment recommended (Google Colab / local CUDA setup)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AaryarajPatil/CNN-Grad-CAM-based-Retina-Disease-Detector.git
   cd CNN-Grad-CAM-based-Retina-Disease-Detector
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the notebook:
   ```bash
   jupyter notebook "notebooks/CNN and Grad-CAM based Multi Retina Disease Detector.ipynb"
   ```

---

## Project Contribution & Responsibilities

Key responsibilities included:
- Dataset acquisition, merging, curation, and preprocessing
- Designing and training the transfer learning model with MobileNetV2
- Model evaluation, accuracy tracking, and error analysis
- Implementing Grad-CAM gradient backpropagation for visual explainability

---

## Future Roadmap

- [ ] Fine-tuning top layers of MobileNetV2 backbone for higher sensitivity
- [ ] Incorporating multi-label classification for co-occurring ocular conditions
- [ ] Integrating tabular patient demographic data with multimodal architectures
- [ ] Exporting lightweight TFLite model for mobile and edge deployment

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Disclaimer

This software is developed strictly for **academic, educational, and research purposes**. It is not certified for clinical diagnostic use.

