import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import librosa
from keras.layers import Dropout, Conv2D, Conv2DTranspose
from keras.models import Model, Sequential
from scipy.io import wavfile
from pathlib import Path
from autoencoder.model import SAMPLE_RATE, HOP_LEN, WINDOW_LEN, WTYPE, preprocess_input

class DropoutAtInference(Dropout):
    # Adding dropout without training does not work
    # See: https://keras.io/api/layers/regularization_layers/dropout/ for details
    def call(self, inputs):
        # Hardcode training to True
        return super().call(inputs, training=True)

def window_audio(y, window_size):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    # Split audio into windows
    windows = []
    for i in range(0, len(y), window_size):
        window = y[i:i+window_size]
        if len(window) < window_size:
            window = np.pad(window, (0, window_size - len(window)))
        windows.append(window)
    return windows

def postprocess_output(S_output, window_length, hop_length, wtype):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    # Remove batch dimension
    S_output = tf.squeeze(S_output, axis=0)

    # Convert to numpy array
    S_output = S_output.numpy()

    # Last dimension is separated into magnitude and phase
    # Reconstruct complex numbers
    S_output = S_output[..., 0] + 1j * S_output[..., 1]

    # Transpose to match librosa format
    S_output = S_output.transpose()

    # Reconstruct audio from output
    y_output = librosa.istft(S_output, n_fft=window_length, hop_length=hop_length, window=wtype)

    # Normalize audio
    y_output = librosa.util.normalize(y_output)
    return y_output

def fog(model, magnitude):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    for i, layer in enumerate(model.layers):
        # Only run fog on Conv2Dlayers; may do others later
        if isinstance(layer, Conv2D) or isinstance(layer, Conv2DTranspose):
            weights = layer.get_weights()
            new_weights = []
            for weight_matrix in weights:
                # Introduce random noise or zero out weights
                mask = np.random.binomial(1, p=0.5, size=weight_matrix.shape) # don't modify every weight
                shape = weight_matrix.shape
                noise = np.random.normal(loc=0.0, scale=magnitude, size=shape)
                new_weight_matrix = weight_matrix + (noise * mask)
                new_weights.append(new_weight_matrix)
            model.layers[i].set_weights(new_weights)
    return model

def lapse(model, magnitude):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    new_model = Sequential()

    for i, layer in enumerate(model.layers):
        new_model.add(layer)
        if isinstance(layer, Conv2D) or isinstance(layer, Conv2DTranspose) and i % 2 == 0:
            new_model.add(DropoutAtInference(magnitude))

    new_model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    return new_model

def floss_model(model, magnitude=0.1, action='fog'):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    if not any(isinstance(layer, Conv2D) for layer in model.layers):
        logging.warn('Model does not contain any Conv2d layers; model will remain unmodified')
        return model
    if action == 'fog':
        model = fog(model, magnitude)
    elif action == 'lapse':
        model = lapse(model, magnitude)
    else:
        logging.warn('Invalid action; model will remain unmodified')
    return model

def run_senseflosser(model_file, magnitude, action, input_file, output_dir, duration, titrate, save_model):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    sample_rate = SAMPLE_RATE
    window_len = WINDOW_LEN
    hop_len = HOP_LEN
    wtype = WTYPE
    orig_model = keras.models.load_model(model_file)

    # Obtain normal output
    y, sr = librosa.load(input_file, mono=True)

    # Model params
    if duration is None:
        try:
            # This is ugly :(
            duration = int(model_file.stem.split('s')[0])
        except ValueError: # If model file name doesn't have a duration in it
            logging.error('Duration must be specified if model file name does not contain duration')
            sys.exit(1)

    sequence_length = duration * sample_rate - ((duration * sample_rate) % window_len)
    freq_bins = window_len // 2 + 1
    windows = ((sequence_length - window_len) // hop_len) + 1
    
    # Preprocess input
    y_proc = preprocess_input(sequence_length, windows, freq_bins, y, sr)

    # Predict
    S_normal_output = orig_model.predict(y_proc)
    normal_output = postprocess_output(S_normal_output, window_len, hop_len, wtype)

    # Get flossin'!
    flossed_outputs = dict()
    for i, m in enumerate(magnitude):
        flossed_model = floss_model(orig_model, magnitude[i], action)
        flossed_output = flossed_model.predict(y_proc)
        flossed_outputs[magnitude[i]] = postprocess_output(flossed_output, window_len, hop_len, wtype)

    # Write waveforms
    output_files = []
    output_dir.mkdir(exist_ok=True)
    output_file_prefix = input_file.stem
    normal_file = output_dir.joinpath(f'{output_file_prefix}_normal.wav')
    wavfile.write(normal_file, sample_rate, normal_output)
    output_files.append(normal_file)
    for m in flossed_outputs:
        flossed_file = output_dir.joinpath(f'{output_file_prefix}_{action}_{m}.wav')
        wavfile.write(flossed_file, sample_rate, flossed_outputs[m])
        output_files.append(flossed_file)

    # Save flossed model
    if save_model:
        if titrate:
            logging.error('Not saving titrated models; please specify a single magnitude.')
            sys.exit(0)
        model_dir = Path('./models')
        model_dir.mkdir(exist_ok=True)
        output_model_prefix = model_file.stem
        flossed_model.save(model_dir.joinpath(f'{output_model_prefix}_{action}_{magnitude[0]}.h5'))
    
    return output_files