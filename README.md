# Plant Disease Detection 


Plant Disease Detection System using Convolutional Neural Networks (CNN) and transfer learning. Specifically, this system leverages the **ResNet-50** architecture implemented in **PyTorch** and has achieved an accuracy of **82% on the validation set**.

## Overview

The Plant Disease Detection System aims to classify images of plant leaves based on the presence and type of disease. This dataset provides labeled images of healthy and diseased plant leaves to facilitate training and testing of machine learning models for plant disease classification.

## Model and Training

- **Model Architecture**: ResNet-50, a powerful deep convolutional neural network pre-trained on ImageNet, was used for transfer learning to leverage the features learned from a large image dataset.
- **Framework**: PyTorch
- **Accuracy**: Achieved 82% accuracy on the validation set.

### Training Details

1. **Data Preprocessing**: Images were resized and normalized to be compatible with ResNet-50’s input requirements.
2. **Transfer Learning**: The model was fine-tuned on this dataset by adapting the final layers of ResNet-50 to classify plant diseases.
3. **Hyperparameters**:
   - Optimizer: SGD or Adam
   - Learning Rate: Adjusted during training
   - Epochs: Chosen based on performance on the validation set

## Dataset Structure

The dataset consists of images organized by folders, each corresponding to a different class (e.g., healthy, various types of plant diseases). Each class has a sufficient number of labeled samples to support model training and validation.


## Usage

1. **Dataset Preparation**: Download and prepare the dataset as per the structure shown above.
2. **Training the Model**: Use the dataset to fine-tune the ResNet-50 model in PyTorch.
3. **Evaluation**: The model can be evaluated on the validation set, where it has achieved an accuracy of 82%.

## Requirements

- Python >= 3.6
- PyTorch >= 1.6
- Additional libraries: `torchvision`, `numpy`, `matplotlib`, `scikit-learn`

Install the requirements via pip:

```bash
pip install torch torchvision numpy matplotlib scikit-learn


