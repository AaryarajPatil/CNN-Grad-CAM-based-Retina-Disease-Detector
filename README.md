# CNN & Grad-CAM based Retina Disease Detector

This repository contains our EDI project focused on Retina Disease
Classification using deep learning and explainable AI techniques.

The model classifies retinal images into:
- Diabetic Retinopathy
- Glaucoma
- Normal

## Dataset

Two publicly available retinal image datasets were collected and combined
to create a custom dataset for this project.

Data preparation included:
- Cleaning and filtering low-quality images
- Removing duplicate samples
- Class-wise organization
- Standard preprocessing

Due to licensing and privacy constraints, the dataset is not included here.

## Approach

- Transfer learning using MobileNetV2 (pretrained on ImageNet)
- CNN-based classification head
- Proper train–validation split
- Grad-CAM for model interpretability

## Results

- Trained for 10 epochs
- Achieved ~87% validation accuracy
- No major signs of overfitting

## My Role

I was primarily responsible for:
- Dataset collection, combination, and cleaning
- Model training and validation
- Performance evaluation
- Explainability using Grad-CAM

## Future Work

- Fine-tuning backbone layers
- Dataset expansion
- Applying ML techniques to structured (tabular) datasets

## Disclaimer

For academic and learning purposes only. Not for clinical use.
