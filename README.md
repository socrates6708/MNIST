# MNIST Digit Classification

## Overview
This script implements a convolutional neural network (CNN) in PyTorch for classifying images from the MNIST dataset, which consists of grayscale images of handwritten digits (0-9).

## Requirements
- Python 3.x
- PyTorch
- Torchvision
- Numpy
  
Install the dependencies using:
```bash
pip install numpy


```
## Running the Model
To train and evaluate the model, use the following command<br>
To evaluation the model, please turn on the flag in the main function to evaluation mode=True
```bash

python mnist.py

```
## Command-line argumenmts for training
This script accepts several optional arguments that you can customize to control the training process:<br>
this script accepts optional arguments for training process
```bash

--epochs: Sets the number of training epochs. The default is 10.
--batch_size: Specifies the batch size for training. The default is 64.
--learning_rate: Determines the learning rate for the optimizer. The default is 0.001.
```

## Model Architecture

The neural network implemented in this script is a convolutional neural network (CNN) for classifying MNIST digits. Here’s a detailed breakdown of its layers and operations:

1. **Convolutional Layers**:
   - `conv1`: A 2D convolutional layer with 32 filters, each of size 3x3, followed by a ReLU activation.
   - `conv2`: A 2D convolutional layer with 64 filters, each of size 3x3, followed by a ReLU activation.
   - `conv3`: A 2D convolutional layer with 128 filters, each of size 3x3, followed by a ReLU activation.

2. **Flattening Layer**:
   - The output from the final convolutional layer is flattened to prepare it for the fully connected layer. This is done by reshaping the tensor using `view(-1, fc_input_size)` where `fc_input_size` is calculated based on the input dimensions.

3. **Fully Connected Layer**:
   - A fully connected (linear) layer that takes the flattened output from the convolutional layers and maps it to 10 output classes (corresponding to the digits 0-9).

4. **Log-Softmax Activation**:
   - The final layer applies a log-softmax activation to produce log-probabilities for each of the 10 classes.

### Forward Pass
During the forward pass:
- The input gray image (with 1 channel) passes through three convolutional layers, transforming the channel dimensions as follows:
  - Input: 1 channel → Conv1: 32 channels
  - Conv1 output: 32 channels → Conv2: 64 channels
  - Conv2 output: 64 channels → Conv3: 128 channels
- The output from the last convolutional layer is flattened and passed through the fully connected layer.
- The final output is obtained by applying the log-softmax function to the result from the fully connected layer.