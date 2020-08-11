# GAN-framework
## Introduction
This repository serves as an easy to use DCGAN framework. Simply download the image dataset you'd like to use and follow the steps below, and the program will be making its own unique images in no time! Note: the quality of the images produced will depend on the dataset fed to it and the hyperparameters used. I discuss some of my results with the program below.

This code is an edited version of pytorch's DCGAN tutorial, which can be found [here](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
## Dependencies
You'll just need numpy, pytorch, and matplotlib.
## Dataset
The dataset used for this program should simply be a directory of images. The directory can also contain as many directories containing images as you please, however the GAN will not consider images in different directories any differently. Also, I am not sure if a directory in a directory in a directory will break the program, as I am not very familar with how pytorch's ``dataloader`` works. The images can be of any size initially (even different sizes), as the program uses pytorch to resize them all to the desired output image size.
## Hyperparameters
Hyperparameters change certain aspects of the networks, optimizers, and data. ``hyperparameters.py`` is a python file with a collection of variables used by ``training.py`` and ``agents.py``. The file has a large commented section explaining what each hyperparameter does. Most importantly, the first line is where you'll input the path to your dataset as a string. This should be the only file you need to touch if you want to create GAN pictures without any coding at all.
## Training
Run this once your hyperparameters have been set. It'll create a direcotry within the project directory called ``results`` and, for every ``sample_interval`` number of batches that the discriminator and generator train on, the program will:

1) 
2) Create a directory within the ``results`` directory named ``[batches_done]epoch[epoch]``, which serves to record the number of batches the nets have trained on and the number of epochs that have occured up to that point, respectively
3) Within this directory, saves a sample of 25 (or if ``batch_size < 25``, then all) of the images created by the generator during that training cycle
