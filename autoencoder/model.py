import sys
import logging
import tensorflow as tf
from keras.layers import (Conv2D,
                         Conv2DTranspose,
                         BatchNormalization,
                         Layer)
from keras.models import Sequential
from keras.optimizers import Adam
import librosa
import numpy as np

# Constants

# Audio
# We have preprocessed the data to only be 22kHz
SAMPLE_RATE= 22050
WINDOW_LEN = 2048
HOP_LEN = WINDOW_LEN // 2
WTYPE = tf.signal.hann_window

# Model
EPOCHS = 50
BATCH_SIZE = 16
SHUFFLE_SIZE = 128

# Dataset
VAL_RATIO = 0.4

def preprocess(audio, sequence_length, windows, freq_bins):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)

    # Remove extra dimensions
    if len(audio.shape) != 1:
        audio = tf.reshape(audio, [audio.shape[0]])

    # Check for and replace Nans
    audio = tf.where(tf.math.is_nan(audio), tf.zeros_like(audio), audio)

    # Check for incorrect audio length
    offset = sequence_length - audio.shape[0]
    if offset > 0:
        logging.warning(f'Audio is too short, padding with {offset} zeros.')
        audio = tf.pad(audio, [[0, offset]], "CONSTANT")
    if offset < 0:
        logging.warning(f'Audio is too long, truncating to {sequence_length} samples.')
        audio = audio[:sequence_length]

    # Normalize audio between -1 and 1
    max_val = tf.reduce_max(tf.abs(audio), axis=0, keepdims=True)
    max_val = tf.maximum(max_val, 1e-5)
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
    if S_audio.shape != tf.TensorShape([windows, freq_bins, 2]):
        logging.warning(f'Audio shape {S_audio.shape} does not match expected shape {[windows, freq_bins, 2]}')
        S_audio = tf.reshape(S_audio, [windows, freq_bins, 2])
    return (S_audio, S_audio)

def load_data(data_dir, sequence_length, windows, freq_bins, percentage=0.6):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    logging.warning('This function expects all audio to be mono, 22 kHz, 16-bit.') 
    logging.warning('If this is not the case, you will likely get strange results.')
    logging.warning('See the ffmpeg preprocessing scripts under ./scripts to preprocess audio.')
    # More efficient than using librosa and iterating over directories
    full_dataset = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=None,
        seed=0,
        labels=None,
        label_mode=None,
        output_sequence_length=sequence_length,
        subset=None)

    dataset_size = full_dataset.cardinality().numpy()
    use_size = max(int(dataset_size * percentage), 1)

    # Use only a percentage of the dataset; full dataset keeps blowing up/exceeding memory
    full_dataset = full_dataset.take(use_size)

    # Sometimes audio is empty; get rid of it
    full_dataset = full_dataset.filter(lambda x: tf.size(x) > 0)

    for features in full_dataset.take(5):
        logging.debug(features.shape)

    # Preprocess data
    full_dataset = full_dataset.map(lambda x: preprocess(x, sequence_length, windows, freq_bins),
                                    num_parallel_calls=tf.data.AUTOTUNE)

    # Split into training and validation
    val_size = int(VAL_RATIO * use_size)
    train = full_dataset.skip(val_size)
    full_val = full_dataset.take(val_size)


    # Split into test
    val = full_val.shard(num_shards=2, index=0)
    test = full_val.shard(num_shards=2, index=1)

    # Cache and prefetch
    # Tensorflow says this is more efficient: (https://www.tensorflow.org/tutorials/audio/simple_audio)
    train = train.cache().shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val = val.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test = test.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train, val, test

def get_strides(windows, freq_bins, num_layers, max_stride=8):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    win_strides = []
    freq_strides = []
    for i in range(1, num_layers + 1):
        found_window_stride = False
        found_freq_stride = False
        for j in range(2, max_stride + 1):  # Really shouldn't have strides larger than 8
            if not found_window_stride and windows % j == 0:
                found_window_stride = True
                win_strides.append(j)
                windows = windows // j
            if not found_freq_stride and freq_bins % j == 0:
                found_freq_stride = True
                freq_strides.append(j)
                freq_bins = freq_bins // j
            if found_window_stride and found_freq_stride:
                break
        # Check if we found a stride
        if not found_window_stride:
            win_strides.append(1)
        if not found_freq_stride:
            freq_strides.append(1)

    # Transpose to get window and freq stride pairs
    strides = list(zip(win_strides, freq_strides))
    return strides
    

def build_model(windows, freq_bins, input_dim=2, var_input=False):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)

    strides = get_strides(windows, freq_bins, 2)

    if var_input:
        logging.debug('Using variable input')
        input_layer = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(None, freq_bins, input_dim))
    else:
        logging.debug('Using fixed input')
        input_layer = Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(windows, freq_bins, input_dim))
    
    model = Sequential([
        # Encoder
        input_layer,
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same', strides=strides[0]),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same', strides=strides[1]),
        BatchNormalization(),

        # Bottleneck
        Conv2D(256, (3, 3), activation='relu', padding='same'),

        # Decoder
        Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=strides[1]),
        BatchNormalization(),
        Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=strides[0]),
        BatchNormalization(),
        Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),

        # Output layer
        Conv2D(input_dim, (3, 3), activation='linear', padding='same')
    ])

    model.compile(optimizer=Adam(), loss='mse', run_eagerly=True, metrics=['mae'])
    return model

def preprocess_input(sequence_length, windows, freq_bins, test_input=None, sr=None):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    if test_input is None:
        test_input, _ = librosa.load(librosa.example('brahms'), sr=SAMPLE_RATE, mono=True)
        test = True
    else:
        test = False
        if sr != SAMPLE_RATE:
            test_input = librosa.resample(test_input, orig_sr=sr, target_sr=SAMPLE_RATE)
    # make sure duration is accurate
    if len(test_input) < sequence_length:
        # pad
        test_input = np.pad(test_input, (0, sequence_length - len(test_input)))
    elif len(test_input) > sequence_length:
        test_input = test_input[:sequence_length]
    test_input[len(test_input)//2] = float('nan')
    test_input = tf.convert_to_tensor(test_input)
    # preprocess
    processed_input, _ = preprocess(test_input, sequence_length, windows, freq_bins)
    # Check if any NaNs remain
    contains_nan = tf.reduce_any(tf.math.is_nan(processed_input))
    logging.debug(f'Contains NaN: {contains_nan.numpy()}')
    if not test:
        # Add batch dimension
        processed_input = tf.expand_dims(processed_input, axis=0)
        return processed_input

# Below debugging functions courtesy of Liqian :)
class CheckNan(Layer):
    def __init__(self, **kwargs):
        super(CheckNan, self).__init__(**kwargs)

    def call(self, inputs):
        checked_input = tf.debugging.assert_all_finite(inputs, 'Decoder output contains NaN')
        return checked_input