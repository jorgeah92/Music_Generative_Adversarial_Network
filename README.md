# deep_learning_project
w251 final project - Spring 2021 UC Berkeley

**Members: Jorge Hernandez, Luke Verdi, Tom Nguyen, Hao Wu**

## This repo contains all of the code used and tested for our final project for w251 Spring 2021

The repo is split into various folder to help organize our content. Each folder has its on Readme with descriptions of the content files:
* Gener_songs - contains samples of generated songs from our models
* maestro-v3.0.0 - the Maestro dataset from Google
* Parsed_data - contains code for various ways of parsing the maestro and a saved numpy array of the parsed maestro files
* Past_attempt_models - contains code for the past attempts of model we played with before coming to our final GAN approach

One this main section of the repo, there are 3 files that all contain our final model:
* Final_generator_model.zip - contains a saved trained model of our GAN generator in the h5 format
* gan_training.py - A python script version of our final model used to train our model on AWS 
* image_GAN_midi_Final.ipynb - A notebook version of our final model which we used to explore our model and outputs


## AWS Instance Specifications

An AWS instance was used to train the GAN models for both 1000 epochs with training set size 200,000 and 2000 epoch with training set size 15,000.

Instance used:
* Location of servers - N. California
* Deep Learning AMI (Ubuntu 18.04) Version 42.1 - ami-0f2dda3eb2d4dfba0
* Instance type - g4dn.2xlarge


## NX deployment
The NX deployment for this project pretty straightforward, as we already trained the model that can be loaded on the NX.
* First we need to setup an NGC container, we used the below command to create it
  - sudo docker pull nvcr.io/nvidia/l4t-ml:r32.4.2-py3
* We made separate directory to store the notebooks and connected it to the container environment when running the image via the commends below.
  - mkdir ~/notebooks
  - sudo docker run -it --rm -v $PWD/notebooks:/notebooks --runtime nvidia --network host nvcr.io/nvidia/l4t-ml:r32.4.2-py3
* It then allows to open a jupyter lab server in our NX browser where we ran the notebook in our Jetson environment to load the model and output the music. The notebook we ran can be referred in this repo in the NX deployment folder.
