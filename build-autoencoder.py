import os
import sys
import argparse
import logging
import librosa
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model
from keras.optimizers import Adam

# Constants
# We have preprocessed the data to only be 22.1kHz
SAMPLE_RATE = 22050
# Tried 30 seconds, got a bunch of OOM
DURATION = 2
N_FEATURES = 1
EPOCHS = 20
BATCH_SIZE = 64
UNITS = 128
VAL_RATIO = 0.3

# Model params
timesteps = SAMPLE_RATE * DURATION
input_dim = N_FEATURES

def preprocess(audio):
    # For some reason Functional.call() is complaining about this
    audio = tf.squeeze(audio, axis=-1)  # Remove unnecessary dimensions
    # Check for nans?
    audio = tf.where(tf.math.is_nan(audio), tf.zeros_like(audio), audio)
    return audio

def load_data(data_dir, sample_rate, duration, percentage=0.6):

    # More efficient than using librosa and iterating over directories
    logging.warning('This function expects all audio to be preprocessed at 22.1 kHz.') 
    logging.warning('If this is not the case, you will likely get strange results.')
    full_dataset = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=1,
        seed=0,
        labels=None,
        label_mode=None,
        output_sequence_length=sample_rate * duration,
        subset=None)

    dataset_size = full_dataset.cardinality().numpy()
    use_size = int(dataset_size * percentage)

    # Use only a percentage of the dataset; full dataset keeps blowing up/exceeding memory
    full_dataset = full_dataset.take(use_size)

    # Split into training and validation
    val_size = int(VAL_RATIO * use_size)
    train = full_dataset.skip(val_size)
    full_val = full_dataset.take(val_size)

    # Preprocess
    train = train.map(preprocess, tf.data.AUTOTUNE)
    full_val = full_val.map(preprocess, tf.data.AUTOTUNE)

    # Split into test
    test = full_val.shard(num_shards=2, index=0)
    val = full_val.shard(num_shards=2, index=1)

    # Batch
    train = train.batch(BATCH_SIZE)
    val = val.batch(BATCH_SIZE)
    test = test.batch(BATCH_SIZE)

    return train, val, test


def build_model(timesteps, input_dim):
    # Modified from https://blog.keras.io/building-autoencoders-in-keras.html
    input_layer = Input(shape=(timesteps, input_dim))
    # Long Short-Term Memory (LSTM), a type of recurrent neural network (https://keras.io/api/layers/recurrent_layers/lstm/)
    # Using 128 units in the LSTM layer is (apparently) a common choice for audio data
    encoder = LSTM(UNITS)(input_layer)
    # Repeat the input timesteps times
    repeated = RepeatVector(timesteps)(encoder)
    # return_sequences=True means that the LSTM layer will return the full sequence of outputs for each input
    # This is what we want, audio in == audio out, nothing fancy
    decoder = LSTM(UNITS, return_sequences=True)(repeated)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=Adam(), loss='mse')
    return autoencoder

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing the audio files')
    parser.add_argument('--log', type=str, default='warn', help='Logging level (choose from: critical, error, warn, info, debug)')

    args = parser.parse_args()


    levels = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }

    loglevel = args.log.lower() if (args.log.lower() in levels) else 'warn'
    logging.basicConfig(stream=sys.stderr, level=levels[loglevel], format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    data_dir = Path(args.data_dir)

    # Load audio and split into datasets
    # Not using test atm
    train, val, _ = load_data(data_dir, SAMPLE_RATE, DURATION)

    # look at data to make sure we aren't crazy
    for audio in train.take(1):
        logging.debug(audio.shape)


    # Build the autoencoder
    autoencoder = build_model(timesteps, input_dim)

    # look at it
    logging.debug(autoencoder.summary())

    # Tensorflow says this is more efficient: (https://www.tensorflow.org/tutorials/audio/simple_audio)
    train = train.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val = val.cache().prefetch(tf.data.AUTOTUNE)

    # Train the model
    autoencoder.fit(x=train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=val)

    # Save model
    autoencoder.save('models/audio_autoencoder.h5')
