import os
import sys
import argparse
import logging
import librosa
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (Input, 
                          Layer, 
                          Conv2D, 
                          Conv2DTranspose, 
                          UpSampling2D, 
                          BatchNormalization, 
                          Reshape, 
                          Cropping2D
                            )
from keras.models import Model, Sequential
from keras.optimizers import Adam

# Constants

# Audio
# We have preprocessed the data to only be 22kHz
SAMPLE_RATE= 22050
WINDOW_LEN = 2048
HOP_LEN = WINDOW_LEN // 2
WTYPE = tf.signal.hann_window

# Model
EPOCHS = 25
BATCH_SIZE = 64
SHUFFLE_SIZE = 100

# Dataset
VAL_RATIO = 0.3

def preprocess(audio, sample_rate, duration):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    # For some reason Functional.call() is complaining about this
    audio = tf.reshape(audio, [-1, tf.shape(audio)[1]])

    # Check for nans
    audio = tf.where(tf.math.is_nan(audio), tf.zeros_like(audio), audio)

    # Check for short audio
    target_length = duration * sample_rate
    pad_length = target_length - tf.shape(audio)[1]
    if pad_length > 0:
        extra = pad_length + tf.shape(audio)[1] % WINDOW_LEN
        full_pad_length = pad_length + extra
        audio = tf.pad(audio, [[0, 0], [0, full_pad_length]], "CONSTANT")

    # Normalize audio between -1 and 1
    max_val = tf.reduce_max(tf.abs(audio), axis=1, keepdims=True)
    max_val = tf.maximum(max_val, 1e-5)  # Prevent division by zero
    audio = audio / max_val

    # Convert to STFT
    S_audio = tf.signal.stft(audio, frame_length=WINDOW_LEN, frame_step=HOP_LEN, window_fn=WTYPE)

    # Check for NaNs in STFT
    reals = tf.math.real(S_audio)
    imags = tf.math.imag(S_audio)
    real_nans = tf.math.is_nan(reals)
    imag_nans = tf.math.is_nan(imags)
    has_nan = tf.logical_or(real_nans, imag_nans)

    # Replace NaNs in STFT
    if tf.reduce_any(has_nan):
        logging.warning('NaN values detected in STFT output, and have been replaced with zeros.')
        reals = tf.where(real_nans, tf.zeros_like(reals), reals)
        imags = tf.where(imag_nans, tf.zeros_like(imags), imags)

    # Stack reals and imags
    S_audio = tf.stack([reals, imags], axis=-1)

    # Make sure the shape matches windows, freq_bins, input_dim
    windows = duration * sample_rate // HOP_LEN
    freq_bins = WINDOW_LEN // 2 + 1 # should be constant, we recreate here just in case
    if S_audio.shape[1:] != tf.TensorShape([windows, freq_bins, 2]):
        logging.warning(f'Audio shape {S_audio.shape} does not match expected shape {[S_audio.shape[0], windows, freq_bins, 2]}')
        S_audio = tf.reshape(S_audio, [-1, windows, freq_bins, 2])
    return (S_audio, S_audio)

def load_data(data_dir, sample_rate, duration, percentage=0.6):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    logging.warning('This function expects all audio to be preprocessed at 22 kHz.') 
    logging.warning('If this is not the case, you will likely get strange results.')
    # More efficient than using librosa and iterating over directories
    ideal_sequence_length = sample_rate * duration - ((sample_rate * duration) % WINDOW_LEN)
    full_dataset = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=BATCH_SIZE,
        seed=0,
        labels=None,
        label_mode=None,
        output_sequence_length=ideal_sequence_length,
        subset=None)

    dataset_size = full_dataset.cardinality().numpy()
    use_size = max(int(dataset_size * percentage), 1)

    # Use only a percentage of the dataset; full dataset keeps blowing up/exceeding memory
    full_dataset = full_dataset.take(use_size)

    # Preprocess data
    full_dataset = full_dataset.map(lambda x: preprocess(x, sample_rate, duration), num_parallel_calls=tf.data.AUTOTUNE)

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


def build_model(windows, freq_bins, input_dim=2):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    model = Sequential()

    # Input layer
    input_layer = Input(shape=(windows, freq_bins, input_dim))
    model.add(input_layer)
    
    # Encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2)))

    # Bottleneck
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))

    # Decoder
    model.add(Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))

    # Output layer
    model.add(Conv2D(input_dim, (3, 3), activation='linear', padding='same'))

    # Output layer does not match input shape for some reason
    model.add(Cropping2D(cropping=((0, 1), (0, 3)))) 

    model.compile(optimizer=Adam(), loss='mse')
    return model

# Below debugging functions courtesy of Liqian :)
def test_preprocess_function(sample_rate, duration):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    test_input, _ = librosa.load('data/fma_small/006/006329.wav.wav', sr=sample_rate)
    # make sure duration is accurate
    if len(test_input) < duration * sample_rate:
        # pad
        test_input = np.pad(test_input, (0, duration * sample_rate - len(test_input)))
    elif len(test_input) > duration * sample_rate:
        test_input = test_input[:duration * sample_rate]
    if len(test_input) % WINDOW_LEN != 0:
        test_input = test_input[:-(len(test_input) % WINDOW_LEN)]
    test_input[len(test_input)//2] = float('nan')
    test_input = tf.convert_to_tensor(test_input)
    # Add dims to match dataset
    test_input = tf.reshape(test_input, [1, -1, 1])
    # preprocess
    processed_input = preprocess(test_input, sample_rate, duration)
    # Check if any NaNs remain
    contains_nan = tf.reduce_any(tf.math.is_nan(processed_input[0]))
    logging.debug(f'Processed input: {processed_input[0].numpy()}')
    logging.debug(f'Contains NaN: {contains_nan.numpy()}')

class CheckNan(Layer):
    def __init__(self, **kwargs):
        super(CheckNan, self).__init__(**kwargs)

    def call(self, inputs):
        checked_input = tf.debugging.assert_all_finite(inputs, 'Decoder output contains NaN')
        return checked_input

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing the audio files')
    parser.add_argument('--duration', type=int, default=30, help='Duration of audio (in seconds) to use for training')
    parser.add_argument('--percentage', type=float, help='Percentage of dataset to use')
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
    duration = args.duration
    percentage = args.percentage if args.percentage else 0.6
    logging.debug(f'Duration set to {duration}')

    # Test the preprocess function
    if loglevel == 'debug':
        test_preprocess_function(SAMPLE_RATE, duration)
    
    # Load audio and split into datasets
    # Not using test atm
    train, val, _ = load_data(data_dir, SAMPLE_RATE, duration, percentage)
    
    # look at data to make sure we aren't crazy
    if loglevel == 'debug':
        pass

    # Build the autoencoder
    # Model params
    windows = duration * SAMPLE_RATE // HOP_LEN
    freq_bins = WINDOW_LEN // 2 + 1
    autoencoder = build_model(windows, freq_bins)

    # look at the model
    logging.debug(autoencoder.summary())

    # Train the model
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    history = autoencoder.fit(x=train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=val, callbacks=[early_stop, reduce_lr])

    # Save model
    autoencoder.save(f'models/{duration}s_audio_autoencoder.h5')