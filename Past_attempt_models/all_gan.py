import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D, ZeroPadding1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv1D, UpSampling1D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# Size vector to generate images from
SEED_SIZE = 100

# Configuration
BATCH_SIZE = 32
BUFFER_SIZE = 60000
IMAGE_SHAPE = (256,3,1)  # make sure GAN matches this


# training data read and convert to TF
train_data_midi = np.load('All_Maestro_Parsed.npy')
train_data_midi = train_data_midi.reshape((train_data_midi.shape[0],256,3,1))
train_data_midi_tf = tf.data.Dataset.from_tensor_slices(train_data_midi) \
    .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def build_generator(seed_size, channels):

    model = Sequential()

    model.add(Dense(4*1*256,activation="relu",input_dim=100))
    model.add(Reshape((4,1,256)))

    model.add(UpSampling2D((2,3)))
    model.add(Conv2D(256,kernel_size=5,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D((2,1)))
    model.add(Conv2D(256,kernel_size=5,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D((2,1)))
    model.add(Conv2D(128,kernel_size=5,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D((2,1)))
    model.add(Conv2D(64,kernel_size=5,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D((2,1)))
    model.add(Conv2D(32,kernel_size=5,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D((2,1)))
    model.add(Conv2D(16,kernel_size=5,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))

    # Final CNN layer
    model.add(Conv2D(1,kernel_size=3,padding="same"))
    model.add(Activation("sigmoid"))

    return model

def build_discriminator(image_shape):

    model = Sequential()

    model.add(Conv2D(16, kernel_size=4, strides=1, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(64, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

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
        print (f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss}, {hms_string(epoch_elapsed)}')
        #save_images(epoch,fixed_seed)

    elapsed = time.time()-start
    print (f'Training time: {hms_string(elapsed)}')

# build the generator
generator = build_generator(SEED_SIZE, 1)
noise = tf.random.normal([1,SEED_SIZE])
generated_image = generator(noise, training=False)

# build the discriminator
discriminator = build_discriminator(IMAGE_SHAPE)
decision = discriminator(generated_image)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy()

# define the optimizers
generator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-4,0.5)

# train!
train(train_data_midi_tf, 10)

# save the generator
generator.save("all_midi_generator")
