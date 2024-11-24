# MNIST Digit Classification

## Overview
This script implements a convolutional neural network (CNN) in PyTorch for classifying images from the MNIST dataset for computation-limited edge device, which consists of grayscale images of handwritten digits (0-9).


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
   - `conv1`: A 2D convolutional layer with 32 filters, each of size 3x3, with depth 32 followed by a ReLU activation.
   - `conv2`: A 2D convolutional layer with 64 filters, each of size 3x3, with depth 64 followed by a ReLU activation.
   - `conv3`: A 2D convolutional layer with 128 filters, each of size 3x3, with depth 128 followed by a ReLU activation.

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

# Quantization branch 

## Desciption 
this branch includes the implementation of PTQ (post training quantization) to enhance performance on devices

## Installation 
Clone this branch using
```bash
git clone -b quantization https://github.com/socrates6708/MNIST/tree/quantization
```


## PTQ Process Overview

The Post-Training Quantization process involves several key steps:


[Training] -> [Preparation] -> [Calibration] -> [Conversion] -> [Inference]

    |              |                 |                |                |
    |              |                 |                |                |

Incorporating a simple visualization diagram into a README.md file typically involves using text-based representation or embedding an image. Given the conceptual nature of the Post-Training Quantization (PTQ) process diagram I described, using a Markdown formatted text diagram might be the most straightforward approach. Here’s how you can type this visualization into your README.md for GitHub:

Markdown Text Diagram in README.md
You can create a simple ASCII-style diagram directly in your Markdown file. Here’s how you might write it:


## PTQ Process Overview

The Post-Training Quantization process involves several key steps:

[Training] ---> [Preparation] ---> [Calibration] ---> [Conversion] ---> [Inference]


### Detailed Explanation of Each Step:

- **Training**: Train the model with floating-point precision.
- **Preparation**: Set up the model for quantization by defining quantization configurations.
- **Calibration**: Run a representative dataset through the model to gather statistics necessary for quantization.
- **Conversion**: Convert the model from floating-point to integer precision using the gathered statistics.
- **Inference**: Deploy the quantized model for efficient inference on target hardware.

This diagram represents the sequential flow and dependencies in the PTQ process.

