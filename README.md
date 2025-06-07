# Neural Networks Python

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python implementation of neural networks from scratch, focusing on educational purposes and understanding the
fundamentals of deep learning.

## Project Overview

This project implements a neural network framework from scratch using NumPy. It provides an implementation of feedforward neural networks with backpropagation, designed primarily for the MNIST handwritten digit classification task.

Key features:

- Implementation of neural networks without using deep learning frameworks
- Configurable network architecture with customizable hidden layers
- Training and evaluation utilities
- Visualization of training and validation metrics
- MNIST dataset processing and batching

## Project Structure

```
neural-networks-python/
├── configs/                # Configuration files
│   ├── config.yaml         # Default model and training configuration
│   └── config_lr_1e-4.yaml # Configuration with learning rate 1e-4
├── data/                   # Data processing utilities
│   └── process_data.py     # MNIST data loading and preprocessing
├── model/                  # Neural network model implementation
│   └── neural_networks.py  # Core neural network implementation
├── tests/                  # Unit tests
│   ├── test_accuracy.py    # Tests for accuracy calculation
│   ├── test_cross_entropy_loss.py # Tests for loss function
│   ├── test_neural_networks.py # Tests for neural network model
│   ├── test_optimizer.py   # Tests for optimizer
│   ├── test_process_data.py # Tests for data processing
│   ├── test_relu.py        # Tests for ReLU activation
│   └── test_softmax.py     # Tests for softmax activation
├── utilities/              # Utility functions
│   ├── accuracy.py         # Accuracy calculation
│   ├── cross_entropy_loss.py # Loss function implementation
│   ├── optimizer.py        # Gradient descent optimization
│   ├── relu.py             # ReLU activation function
│   ├── sigmoid.py          # Sigmoid activation function
│   └── softmax.py          # Softmax activation function
├── environment.yaml        # Conda environment specification
├── experiments.ipynb       # Jupyter notebook for experiments
├── generate_visuals.py     # Script for generating visualizations
├── README.md               # Project documentation
└── train.py                # Training script
```

## Neural Network Architecture

The implementation includes:

- A fully-connected neural network with configurable hidden layers
- ReLU activation for hidden layers
- Softmax activation for the output layer
- Cross-entropy loss function
- Simple mini-batch stochastic gradient descent optimizer
- L1 and L2 regularization

The default architecture is designed for MNIST classification:

- Input layer: 784 neurons (28x28 pixel images)
- Hidden layer: 128 neurons with ReLU activation
- Output layer: 10 neurons (digits 0-9) with softmax activation

## Getting Started

### Prerequisites

This project uses Python 3.11 and several dependencies. You can set up the environment using Conda:

```bash
conda env create -f environment.yaml
conda activate nn-p
```

### Running the Training

To train the neural network on the MNIST dataset, write the following lines of code in a python file:

```bash
trainer = Trainer()
trainer.train()
```

The training script will:

1. Load and preprocess the MNIST dataset
2. Initialize the neural network model
3. Train the model for the specified number of epochs
4. Generate plots showing training and validation loss/accuracy

## Implementation Details

### Documentation Standard

All functions and methods in this project follow a standardized docstring format:

```python
"""Brief description of the function.

:param param_name: Description of the parameter
:param param_name2: Description of another parameter
:return: Description of what the function returns (if applicable)
"""
```

This documentation style improves code readability across the project.

### Data Processing

The `process_data.py` module handles:

- Loading the MNIST dataset using scikit-learn
- Splitting data into training and testing sets
- Creating batches for mini-batch gradient descent

### Neural Network Model

The `neural_networks.py` module implements:

- Weight initialization
- Forward pass computation
- Backpropagation for gradient calculation

### Utilities

The utilities directory contains modular implementations of:

- Activation functions (ReLU, Sigmoid)
- Loss function (Cross-entropy)
- Accuracy calculation
- Optimization algorithm (Gradient descent with L1/L2 regularization)

### Visualization

The `generate_visuals.py` module provides functions for:

- Generating heatmaps
- Creating bar graphs
- Visualizing t-SNE clusters of model outputs
- Generating confusion matrices for model evaluation

## Testing

The project includes unit tests for all major components. To run the tests:

```bash
pytest tests/
```

## Acknowledgments

This project is designed for educational purposes to understand the inner workings of neural networks without relying on
deep learning frameworks.
