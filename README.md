# CIFAR-10 Image Classification with MLP

A deep learning project focused on multi-class image classification using a custom-built Multi-Layer Perceptron (MLP) architecture. This notebook leverages the CIFAR-10 dataset and TensorFlow/Keras to train, evaluate, and analyze a neural network classifier.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)

## Overview

This project implements a flexible and modular MLP (Multi-Layer Perceptron) model to classify 60,000 images in the CIFAR-10 dataset. It includes preprocessing steps, model definition, training, evaluation, and result visualization.

## Dataset

- **Dataset**: CIFAR-10
- **Training Samples**: 50,000
- **Test Samples**: 10,000
- **Image Dimensions**: 32x32x3 (RGB)
- **Number of Classes**: 10

## Features

- Full preprocessing pipeline:
  - Normalization (pixel values scaled to [0, 1])
  - One-hot encoding of labels
  - Custom train/validation/test split
- Modular MLP architecture:
  - Configurable hidden layers
  - L2 regularization
  - Batch normalization (optional)
  - Dropout for overfitting control
- Evaluation metrics and model performance visualization

## Model Architecture

The MLP model includes:

- Input: Flatten layer
- Hidden Layers:
  - Dense layers with ReLU activation
  - L2 regularization
  - Optional BatchNormalization
  - Dropout (default 0.5)
- Output: Dense layer with 10 units and softmax activation

Model is implemented using `tf.keras.Sequential`.

## Installation

To run this notebook:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Usage

1. Clone the repository or download the notebook.
2. Open the notebook in Jupyter or a platform like Google Colab/Kaggle.
3. Run all cells to load the dataset, train the model, and see the results.

## Results

The trained model achieves reasonable performance on the CIFAR-10 dataset. Key performance metrics include accuracy and loss over training/validation, which are plotted at the end of the notebook.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- scikit-learn
- Matplotlib
