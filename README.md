# Autoencoder-using-PyTorch

This repository contains a PyTorch implementation of an Autoencoder and a Variational Autoencoder (VAE) for generating new data that's similar to the Fashion MNIST dataset.

## Requirements
- Python 3.x
- PyTorch
- Torchvision
- Numpy
- Matplotlib
- Keras (For loading the Fashion MNIST dataset)
## Installation
Install the required packages using pip:
```bash
pip install torch torchvision numpy matplotlib keras
```
## Usage
Clone this repository and navigate to the project directory. Run the script VAE.ipynb (which should contain the provided code) to train the VAE model on the Fashion MNIST dataset and visualize the generated images.
```bash
git clone https://github.com/ssakhash/Autoencoders-using-PyTorch.git
```
## Overview
This project includes the following steps:

1) Importing Required Libraries:
   - Essential libraries for tensor computation, dataset loading, neural network construction, etc. are imported.
2) Initializing Device Details and Importing the Dataset:
   - Sets up the device (CPU or CUDA) and loads the Fashion MNIST dataset using Keras.
3) Dataset Preprocessing:
   - Scales and reshapes the dataset.
4) Initializing Data Loaders:
   - Train and test data loaders are initialized using PyTorch’s DataLoader class.
5) Defining Variational Autoencoder (VAE) Class:
   - Defines the architecture of the VAE. It consists of an encoder, a reparameterization step, and a decoder.
6) Defining the ELBO Loss Function:
   - The loss function for VAE is defined, which includes reconstruction loss (MSE Loss) and the KL divergence.
7) Training the Autoencoder Model:
   - The VAE model is trained using the SGD optimizer.
8) Evaluating Reconstruction Capabilities on Test Set:
   - The model's reconstruction capabilities are evaluated using samples from the test set.
9) Plotting Graph of Loss Versus Epochs:
   - A graph is plotted to visualize the lowest loss per epoch over time.

## Model Architecture
The variationalAE class defines the VAE. It has two main parts:
- Encoder: Maps the input data to a latent space. It is implemented as a neural network with one hidden layer.
- Decoder: Maps points in the latent space back to the data space (i.e., reconstructs the data). It is also implemented as a neural network with one hidden layer.
The reparameterization trick is used to allow backpropagation through the stochastic latent space.

## Results
After training, the model's reconstruction capabilities are tested on samples from the test set. The script will output two rows of images: the first row contains the original test images, and the second row contains the corresponding reconstructed images.
