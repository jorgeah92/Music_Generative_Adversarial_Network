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