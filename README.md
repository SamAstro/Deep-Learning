# Udacity Deep Learning Nanodegree Foundation Program

## Synopsis

This is the repository of all the projects I will carry out during [Udacity Deep
Learning Nanodegree Foundation
Program](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101)

## Table of contents
### Project 1 -- your first neural network
Using only numpy, we build a neural network from scratch to carry out
predictions on daily bike rental ridership.

### Project 2 -- images classification
Using tensorflow, we build a convolutional neural network to classify images
from the CIFAR-10 dataset. The dataset consists of airplanes, dogs, cats, and
other objects.

My network is as follows:
* 3 convolutionnal and max pool layers
* 1 flatten layer
* 3 fully connected layers with 0.5 dropout
* 1 output layer

Epoch: 20
Batch size = 128

### Project 3 -- TV Script Generation
In this project, we'll generate our own
[Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.
We'll be using part of the [Simpsons
dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts
from 27 seasons.  The Neural Network we'll build will generate a new TV script
for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern).

My hyperparameters:
* num_epochs = 100  
* batch_size = 64  
* rnn_size = 512  
* seq_length = 16  
* learning_rate = 0.001  
* show_every_n_batches = 128

### Project 4 -- Language Translation
In this project, you’re going to take a peek into the realm of neural network
machine translation. You’ll be training a sequence to sequence model on a
dataset of English and French sentences that can translate new sentences from
English to French.

My hyperparameters:
* epochs = 4  
* batch_size = 256  
* rnn_size = 128  
* num_layers = 2  
* encoding_embedding_size = 50  
* decoding_embedding_size = 50  
* learning_rate = 0.005  
* keep_probability = 0.7

## Prerequisites
You need:

* Jupyter  
* Python 3.x  
* The following Python library:  
    * Numpy
    * Matplotlib
    * Tensorflow and all its dependencies

## Running

To run the notebook, you need Jupyter installed with a Python 3.x kernel. Then
just run 
```
> jupyter-notebook-3.x notebook_name.ipynb
```
in the corresponding directory.
