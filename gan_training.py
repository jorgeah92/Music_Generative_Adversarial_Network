from numpy.lib.format import BUFFER_SIZE
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D,ZeroPadding1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv1D,UpSampling1D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
from tqdm import tqdm
import os 
import time
import matplotlib.pyplot as plt
import pandas as pd
from mido import MidiFile, MidiTrack, Message
from music21 import *
from IPython.display import Image

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def recreate_midi(df_first_notes, speed=20000):
    # function to take a dataframe created by something like parse_notes() or a gan and return a midi
    
    # Can start by reverse scaling the note:
    df_reversed = df_first_notes.copy()
    df_reversed['note'] = round(df_reversed['note'] * 88 + 20)  # might want to have something more special than round()
    df_reversed.note = df_reversed.note.astype(int)
    df_reversed['velocity'] = 60  # create a uniform middling velocity

    # recreate the absolute time index and drop time_since_last (we'll recreate it with the stop signals)
    df_reversed['time_index'] = df_reversed.time_since_last.cumsum()
    df_reversed = df_reversed.drop(columns = 'time_since_last')

    # create a stop signal for each note at the appropriate time_index:
    for i in range(len(df_reversed)):
        stop_note = pd.DataFrame([[df_reversed.note[i], 0, 0, df_reversed.duration[i] + df_reversed.time_index[i]]],
                                 columns=['note', 'duration', 'velocity', 'time_index'])
        df_reversed = df_reversed.append(stop_note, ignore_index=True)
    df_reversed = df_reversed.sort_values('time_index').reset_index(drop=True)

    # recreate time_since last with the stop note signals
    df_reversed['time'] = [0] + [df_reversed.time_index[i+1] - df_reversed.time_index[i] 
                                 for i in range(len(df_reversed)-1)]
    # and now we don't need duration or time_index so can drop those
    df_reversed = df_reversed.drop(columns = {'time_index','duration'})

    # finally, we need to scale the time since last note appropriately:
    df_reversed['time'] = round(df_reversed['time'] * speed)
    df_reversed.time = df_reversed.time.astype(int)

    # finally, recreate the midi and return
    mid_remade = MidiFile()
    track = MidiTrack()
    mid_remade.tracks.append(track)
    track.append(Message('program_change', program=0, time=0))
    for i in range(len(df_reversed)):
        track.append(Message('note_on', note=df_reversed.note[i], velocity=df_reversed.velocity[i], time=df_reversed.time[i]))

    return mid_remade

# For CNNs

def build_generator(seed_size, channels):
    model = Sequential()

    model.add(Dense(4*256, activation="relu",input_dim=SEED_SIZE))
    model.add(Reshape((4,256)))

    model.add(UpSampling1D())
    model.add(Conv1D(256,kernel_size=2,padding="same"))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation("relu"))

    model.add(UpSampling1D())
    model.add(Conv1D(128,kernel_size=2,padding="same"))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation("relu"))

    # Output resolution, additional upsampling
    model.add(UpSampling1D())
    model.add(Conv1D(64,kernel_size=2,padding="same"))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation("relu"))

    model.add(UpSampling1D())
    model.add(Conv1D(32,kernel_size=2,padding="same"))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation("relu"))

    model.add(UpSampling1D(size=3))
    model.add(Conv1D(16,kernel_size=2,padding="same"))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Activation("relu"))

    # Final CNN layer
    model.add(Conv1D(3,kernel_size=2,padding="same"))
    model.add(Activation("tanh"))
    
    return model

def build_discriminator(image_shape):
    model = Sequential()

    model.add(Conv1D(16, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv1D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding1D(padding=((0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv1D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding1D(padding=((0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv1D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv1D(256, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv1D(512, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    seed = tf.random.normal([BATCH_SIZE, SEED_SIZE])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(seed, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    

        gradients_of_generator = gen_tape.gradient(\
            gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(\
            disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(
            gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(
            gradients_of_discriminator, 
            discriminator.trainable_variables))
    return gen_loss,disc_loss

def train(dataset, epochs):
    #fixed_seed = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, 
    #                                   SEED_SIZE))
    start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        gen_loss_list = []
        disc_loss_list = []

        for image_batch in dataset:
            t = train_step(image_batch)
            gen_loss_list.append(t[0])
            disc_loss_list.append(t[1])

        g_loss = sum(gen_loss_list) / len(gen_loss_list)
        d_loss = sum(disc_loss_list) / len(disc_loss_list)

        epoch_elapsed = time.time()-epoch_start
        print (f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss},'\
           ' {hms_string(epoch_elapsed)}')
        #save_images(epoch,fixed_seed)
        if epoch // 100 == 0:
            generator.save(os.path.join("/home/ubuntu/","long_generator" + str(epoch) + ".h5"))
        else:
            continue

    elapsed = time.time()-start
    print (f'Training time: {hms_string(elapsed)}')

###########################################################################
# PARAMETERS

CHANNELS = 3

# Size vector to generate images from
SEED_SIZE = 100

# Configuration
BATCH_SIZE = 64
EPOCH = 2000
BUFFER_SIZE = 60000
#############################################################################

training_data_midi = np.load('All_Maestro_Parsed.npy')

# training_data_midi = np.float32(training_data_midi)

train_dataset_midi = tf.data.Dataset.from_tensor_slices(training_data_midi[:,:192,:]) \
    .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

image_shape = (192,CHANNELS)

generator = build_generator(SEED_SIZE, CHANNELS)
discriminator = build_discriminator(image_shape)

generator_optimizer = tf.keras.optimizers.Adam(1.0e-4,0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(7.5e-5,0.5)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy()

train(train_dataset_midi, EPOCH)


noise = tf.random.normal([1, SEED_SIZE])
generated_song = generator(noise, training=False)
decision = discriminator(generated_song)
print(decision)

midi_dat = generated_song.numpy()

midi_df = pd.DataFrame(midi_dat[0],columns=["note", "duration",'time_since_last'])
mid_made = recreate_midi(midi_df)
mid_made.save('song_gen.mid')

generator.save(os.path.join("/home/ubuntu/","long_generator.h5"))