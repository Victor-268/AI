# Traffic Sign Recognition

## Overview
This project implements a traffic sign recognition system using a Convolutional Neural Network (CNN) in TensorFlow. The dataset used contains images of 43 different types of traffic signs.

## Data
The dataset is organized into 43 categories, each containing images of a particular type of traffic sign. Each category is represented by a directory named after its category number (0-42).

## Model Architecture
The CNN model consists of:
- Three convolutional layers with 32, 64, and 128 filters respectively, each followed by a max-pooling layer.
- A flattening layer to convert the 2D matrix to a 1D vector.
- A dense hidden layer with 512 units and ReLU activation.
- A dropout layer to prevent overfitting.
- An output layer with softmax activation to classify the images into one of the 43 categories.

## Training
The model is trained using the Adam optimizer and categorical cross-entropy loss for 10 epochs. The data is split into training and testing sets with a 60-40 ratio.

## Results
The model achieves satisfactory accuracy on the test set. The performance can be further improved by tuning hyperparameters, adding more data augmentation, or experimenting with different network architectures.

## Experiments
- Initial experiments with fewer filters and layers showed lower accuracy.
- Adding more convolutional layers and increasing the number of filters improved the accuracy.
- Introducing dropout helped in reducing overfitting.
