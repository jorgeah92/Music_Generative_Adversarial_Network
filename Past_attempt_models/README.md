# Past attempts folder

This folder contains code used for past approaches to testing the generation of piano music using the Maestro dataset.

* gan_piano.py - a modified version of the Pokemon article GAN code. This file was tweaked to enable it to train with the maestro dataset
* piano_lstm.py - a modified version fo the Pokemon article LSTM code. Modified to train with the Maestro dataset
* gan_luke.py - a modified version of gan_piano.py to try to incorporate the main parsing approach used with our image GAN attempt
* all_gan.py - an attempt to use all data and a 256x3 representation with conv2D for generation.
* all_gan_square.py - an attempt to use all data and a square representation like 16x16x3 or 32x16 with conv2D for generation.
* start-gan-1D.py - early experimentation with start of songs and conv1D
* start-gan-2D.py - experimentation with start of songs and conv2D using various activations, shapes, etc. Similar to all_gan_square and used to decide what to try to run there.

**Pokemon article link:** https://towardsdatascience.com/generating-pokemon-inspired-music-from-neural-networks-bc240014132