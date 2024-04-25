import os
import sys
import argparse
import logging
import librosa
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from keras.layers import Layer
from keras.layers import Input, LSTM, RepeatVector, Dense
from keras.models import Model
from keras.optimizers import Adam

# Constants
# We have preprocessed the data to only be 22.1kHz
SAMPLE_RATE = 22050
# Default 30 seconds, can overwrite
# Don't recommend using 30 seconds anyways, you will get a bunch of OOM
DURATION = 30
N_FEATURES = 1
EPOCHS = 20
BATCH_SIZE = 8
SHUFFLE_SIZE = 100
UNITS = 128
VAL_RATIO = 0.3

# Model params
timesteps = SAMPLE_RATE * DURATION
input_dim = N_FEATURES

def preprocess(audio):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    # For some reason Functional.call() is complaining about this
    audio = tf.squeeze(audio, axis=-1)  # Remove unnecessary dimensions
    audio = tf.expand_dims(audio, axis=-1) # last dim was NAN, expand it to match the shape 
    # Check for nans?
    audio = tf.where(tf.math.is_nan(audio), tf.zeros_like(audio), audio)
    # Check for nans after process
    nan_mask = tf.math.is_nan(audio)
    if tf.reduce_any(tf.math.is_nan(audio)):
        audio = tf.where(nan_mask, tf.zeros_like(audio), audio)
        if tf.reduce_any(tf.math.is_nan(audio)):
            logging.error('NaNs remain after attempting to replace them.')
            logging.warning('Replacing audio with 0s.')
            audio = tf.zeros_like(audio)
    return (audio, audio)

def load_data(data_dir, sample_rate, duration, percentage=0.6):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    logging.warning('This function expects all audio to be preprocessed at 22.1 kHz.') 
    logging.warning('If this is not the case, you will likely get strange results.')
    # More efficient than using librosa and iterating over directories
    full_dataset = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=BATCH_SIZE,
        seed=0,
        labels=None,
        label_mode=None,
        output_sequence_length=sample_rate * duration,
        subset=None)

    dataset_size = full_dataset.cardinality().numpy()
    use_size = int(dataset_size * percentage)

    # Use only a percentage of the dataset; full dataset keeps blowing up/exceeding memory
    full_dataset = full_dataset.take(use_size)

    # Preprocess data
    full_dataset = full_dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Split into training and validation
    val_size = int(VAL_RATIO * use_size)
    train = full_dataset.skip(val_size)
    full_val = full_dataset.take(val_size)


    # Split into test
    val = full_val.shard(num_shards=2, index=0)
    test = full_val.shard(num_shards=2, index=1)

    # Cache and prefetch
    # Tensorflow says this is more efficient: (https://www.tensorflow.org/tutorials/audio/simple_audio)
    train = train.cache().shuffle(SHUFFLE_SIZE).prefetch(tf.data.AUTOTUNE)
    val = val.cache().prefetch(tf.data.AUTOTUNE)
    test = test.cache().prefetch(tf.data.AUTOTUNE)

    return train, val, test


def build_model(timesteps, input_dim):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    # Modified from https://blog.keras.io/building-autoencoders-in-keras.html
    input_layer = Input(shape=(timesteps, input_dim))
    # Long Short-Term Memory (LSTM), a type of recurrent neural network (https://keras.io/api/layers/recurrent_layers/lstm/)
    # Using 128 units in the LSTM layer is (apparently) a common choice for audio data
    encoder = LSTM(UNITS)(input_layer)
    # Repeat the input timesteps times
    repeated = RepeatVector(timesteps)(encoder)
    # return_sequences=True means that the LSTM layer will return the full sequence of outputs for each input
    # This is what we want, audio in == audio out, nothing fancy
    decoder = LSTM(input_dim, return_sequences=True)(repeated)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse', run_eagerly=True)
    return autoencoder

# Below debugging functions courtesy of Liqian :)
def test_preprocess_function():
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    # a test tensor with NaN values
    test_input = tf.constant([[1.0], [float('nan')], [3.0]])
    # preprocess
    processed_input = preprocess(test_input)
    # Check if any NaNs remain
    contains_nan = tf.reduce_any(tf.math.is_nan(processed_input[0]))
    logging.debug(f'Processed input: {processed_input[0].numpy()}')
    logging.debug(f'Contains NaN: {contains_nan.numpy()}')

def check_shape(audio):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    assert audio.shape[1:] == (SAMPLE_RATE * DURATION, 1), f'Wrong shape: {audio.shape}'
    return audio

class CheckNan(Layer):
    def __init__(self, **kwargs):
        super(CheckNan, self).__init__(**kwargs)

    def call(self, inputs):
        checked_input = tf.debugging.assert_all_finite(inputs, 'Decoder output contains NaN')
        return checked_input

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing the audio files')
    parser.add_argument('--duration', type=int, help='Duration of audio (in seconds) to use for training')
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
    DURATION = args.duration

    # Load audio and split into datasets
    # Not using test atm
    train, val, _ = load_data(data_dir, SAMPLE_RATE, DURATION)
    
    # look at data to make sure we aren't crazy
    for audio in train.take(5):
        logging.debug(audio[0].shape)

    # Build the autoencoder
    autoencoder = build_model(timesteps, input_dim)

    # test the model with a random tensor
    test_audio = tf.random.normal([64, 44100, 1])
    
    # Test the preprocess function
    test_preprocess_function()

    # look at the model
    logging.debug(autoencoder.summary())

    # Train the model
    history = autoencoder.fit(x=train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=val)

    # Save model
    autoencoder.save(f'models/{DURATION}s_audio_autoencoder.h5')
