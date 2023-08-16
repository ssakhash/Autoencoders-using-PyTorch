# Autoencoder and Variational Autoencoder using PyTorch

This repository contains a PyTorch implementation of an Autoencoder and a Variational Autoencoder (VAE) for generating new data that's similar to the Fashion MNIST dataset.

## Theory
### Autoencoders
Autoencoders are a type of artificial neural network used to learn efficient representations of data, typically for the purpose of dimensionality reduction or feature learning. An autoencoder network is typically trained to map input data into a compressed representation and then map it back to the original input data. It consists of two main parts: the encoder, which compresses the input into a latent-space representation, and the decoder, which maps the encoded data back to the original data space. Ideally, an autoencoder will learn to encode useful attributes of the input data in the compressed representation, effectively learning to encode the most important features of the data without supervision.

### Variational Autoencoders (VAEs)
Variational Autoencoders (VAEs) are a type of autoencoder with added constraints on the encoded representations being learned. Unlike a traditional autoencoder, which maps the input data to a single point in a latent space, VAEs map the input data to a distribution in the latent space. This is usually done by having the encoder network predict a mean and variance that define a Gaussian distribution representing the data’s distribution in the latent space. The core idea behind VAEs is the introduction of a probabilistic, generative model that learns to map inputs to a probabilistic distribution in the latent space.

### Why are Variational Autoencoders considered better than Autoencoders?
One of the key advantages of VAEs over traditional autoencoders is this ability to generate new data that's similar to the training data. Since VAEs are generative models that learn the distribution of the data in a latent space, they can generate new data points that are likely under this distribution, effectively allowing for data generation. VAEs also enforce a structured, continuous distribution in the latent space, which ensures that similar data points are mapped close to each other in this space. This makes VAEs very useful for tasks where the latent space needs to have a meaningful structure, such as generative art creation, image generation, semi-supervised learning, and anomaly detection.

Additionally, VAEs integrate probabilistic reasoning, allowing us to model uncertainty and variability in the data, which can be crucial for many real-world applications. They formalize the process of data generation as a probabilistic process and provide a principled framework under which the latent space is regularized during training, typically via a term that encourages the learned representations to approximate a prior distribution (usually a standard normal distribution).

In summary, VAEs not only allow us to compress data like autoencoders but, more importantly, they provide a way to generate new, similar data, making them a powerful tool in the generative models category.

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
