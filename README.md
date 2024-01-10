# Image-Classification

## Overview

This project is focused on image classification using Convolutional Neural Networks (CNNs) implemented in TensorFlow and Keras. The goal is to classify images into different categories such as Coast, Forest, Highway, Kitchen, Mountain, Office, Store, Street, and Suburb.


## Table of Contents

- [Dependencies](#dependencies)
- [Models](#models)
  - [Convolutional Neural Network](#convolutional-neural-network)
  - [Transfer Learning with InceptionV3](#transfer-learning-with-inceptionv3)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dependencies

The project relies on the following Python libraries:

- NumPy
- TensorFlow
- Matplotlib
- scikit-learn
- Keras

## Models

### Convolutional Neural Network

This CNN model consists of 14 layers. The first 12 consist 3 sets of a 2D convolution layer with a stride of 3, a batch normalization layer, a rectified linear unit activation layer, and a 2D max pooling layer. The 13th layer flattens the input from the final max pooling layer and the 14th layer is a fully connected layer with 9 outputs that uses softmax activation. A learning rate of 0.0001, a batch size of 64 were used, the model was trained using mini-batch gradient descent with momentum for 40 epochs.

### Transfer Learning with InceptionV3 

This Transfer Learning model consists of 15 layers. The first 13 consist 3 sets of a 2D convolution layer with a stride of 3, a batch normalization layer, a rectified linear unit activation layer, and a 2D max pooling layer. After the second set a dropout layer is added with a dropout rate of 50 percent. The 14th layer flattens the input from the final max pooling layer and the 15th layer is a fully connected layer with 9 outputs that uses softmax activation. A learning rate of 0.0001, a batch size of 64 were used, the model was trained using mini-batch gradient descent with momentum for 40 epochs.

## Results

### Convolutional Neural Network

### Transfer Learning with InceptionV3
