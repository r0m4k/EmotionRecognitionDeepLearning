# EEG Emotion Recognition System

> **Advanced emotion classification from EEG signals using CNN-Transformer architecture**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

## Overview

This project implements an **EEG-based emotion recognition system** that classifies human emotions into three categories: **Positive**, **Neutral**, and **Negative** using brain signal data. The system combines Convolutional Neural Networks (CNN) with Transformer architecture to achieve superior performance on emotion classification tasks.

### Key Features

- **Multi-dataset Support**: SEED, SEED-French, and SEED-German datasets
- **Advanced Architecture**: CNN-Transformer hybrid model
- **High Performance**: 92.2% accuracy with 0.92 F1-score
- **Real-time Processing**: Optimized for 1-second EEG windows
- **Comprehensive Evaluation**: Detailed metrics and visualizations

### Performance Results

<div align="center">

**Model Performance Visualization**

<img width="600" alt="Model Performance" src="https://github.com/user-attachments/assets/6974eb40-4b8d-4c83-a7bb-d59fba9b8f5d" />

*Figure 1: Confusion matrix for test set*

</div>

### Training Progress

<div align="center">

**Training and Validation Metrics**

<img width="600" alt="Training Progress" src="https://github.com/user-attachments/assets/f75da091-8e0c-4394-b2f8-32e300ce3854" />

*Figure 2: Training loss and accuracy curves over epochs*

</div>

## Architecture

### Model Components

```
EEG Signal (5 electrodes × T time windows × 5 frequency bands)
    ↓
CNN Feature Extraction (2D Convolution)
    ↓
Linear Embedding (32-64 dimensions)
    ↓
Positional Encoding
    ↓
Transformer Encoder (2-4 layers, 4-8 attention heads)
    ↓
Global Average Pooling
    ↓
Classification Head (3 emotion classes)
```

### Technical Specifications

- **Input**: EEG signals from 5 key electrodes (AF3, AF4, T7, T8, Pz)
- **Frequency Bands**: Delta, Theta, Alpha, Beta, Gamma
- **Time Windows**: Variable length (1-second segments)
- **Output**: 3-class emotion classification

## Dataset Information
<div align="center">
    
<img width="600" alt="3" src="https://github.com/user-attachments/assets/74bc1529-3365-4449-b39d-40c25e6785d6" />

*Figure 3: Dataset distribution*
</div>

| Dataset | Samples | Distribution | Language |
|---------|---------|-------------|----------|
| **SEED** | 675 | Balanced (225 each) | Chinese |
| **SEED-French** | 480 | 144/168/168 | French |
| **SEED-German** | 300 | 80/120/100 | German |
| **Unified** | 1,455 | 449/513/493 | Multi-language |

## Performance Results

### Model Performance
- **Validation Accuracy**: 92.2%
- **F1-Score**: 0.92 (weighted)
- **Best Epoch**: 18
- **Training Time**: ~100 epochs

## Installation & Setup

### Prerequisites
```bash
Python 3.11+
PyTorch 2.0+
NumPy, Pandas, Matplotlib
Scikit-learn
Seaborn
```

## Key Technical Details

### Data Processing Pipeline
1. **Signal Extraction**: 1-second EEG windows from 62 electrodes
2. **Electrode Selection**: Focus on 5 key electrodes (AF3, AF4, T7, T8, Pz)
3. **Frequency Analysis**: 5 frequency bands (δ, θ, α, β, γ)
4. **Normalization**: Standard preprocessing and feature scaling

### Model Architecture Details
- **CNN Layer**: 2D convolution with 8-16 output channels
- **Embedding**: Linear projection to 32-64 dimensions
- **Transformer**: 2-4 encoder layers with 4-8 attention heads
- **Dropout**: 0.1-0.3 for regularization
- **Optimizer**: Adam with learning rate 1e-3 to 5e-4

## Results Visualization

The project includes comprehensive visualizations:
- **Dataset Distribution**: Emotion label distribution across datasets
- **Training Curves**: Accuracy and loss over epochs
- **Confusion Matrix**: Detailed classification results
- **Performance Metrics**: F1-score, precision, recall

## Hyperparameter Tuning

The system includes automated hyperparameter optimization:
- **Grid Search**: 100 model configurations tested
- **Parameters**: Batch size, CNN channels, embedding dimensions
- **Optimization**: Attention heads, transformer layers, learning rate

## Applications

This system can be applied in:
- **Healthcare**: Mental health monitoring and therapy
- **Human-Computer Interaction**: Adaptive interfaces
- **Gaming**: Emotion-aware gaming experiences
- **Research**: Affective computing studies

Warning: dataset is not included due to its large size. Please email me - r.dolgopolyi {at} acg {dot} edu - if you would like to obtain it.
