import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import librosa
from build_autoencoder import SAMPLE_RATE

def window_audio(y, window_size, overlap=0.5):
    # Split audio into windows
    hop_length = int(window_size * overlap)
    windows = []
    for i in range(0, len(y), hop_length):
        window = y[i:i+window_size]
        if len(window) < window_size:
            window = np.pad(window, (0, window_size - len(window)))
        windows.append(window)
    return windows

def preprocess_input(y, sr, model):
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    shape = model.input.shape
    target_len = shape[1]
    if len(y) < target_len:
        y = [np.pad(y, (0,target_len - len(y)))]
    elif len(y) > target_len:
        y = window_audio(y, target_len)
    y_windows = []
    for window in y:
        window = window.reshape(shape[1:])
        window = tf.convert_to_tensor(window)
        # Add batch dimension
        window = tf.expand_dims(window, axis=0)
        y_windows.append(window)
    return y_windows, SAMPLE_RATE

def floss_ltsm(layer, magnitude):
    # Example: Manipulating LSTM layer weights
    weights = layer.get_weights()
    new_weights = []
    for weight_matrix in weights:
        # Introduce random noise or zero out weights
        shape = weight_matrix.shape
        noise = np.random.normal(loc=0.0, scale=magnitude, size=shape)
        new_weight_matrix = weight_matrix * (1 + noise)
        new_weights.append(new_weight_matrix)
    layer.set_weights(new_weights)

    return layer

def floss_dense(layer):
    # Example: Manipulating Dense layer weights
    weights = layer.get_weights()
    new_weights = []
    for weight_matrix in weights:
        # do something here
        pass
    return layer

def floss_model(model, magnitude=0.1):
    if 'LSTM' not in model.summary():
        logging.warn('Model does not contain any LSTM layers; model will remain unmodified')
        return model
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], keras.layers.LSTM):
            model.layers[i] = floss_ltsm(model.layers[i], magnitude)
        elif isinstance(model.layers[i], keras.layers.Dense):
            model.layers[i] = floss_dense(model.layers[i])
        else:
            pass
    return model


            
# more ideas

# something that tries to mimic how vinyl or tape might deteriorate
# slowed/reverbed (but how to translate that into latent space?)

# FWIW, I've been told that LTSM layers are quite sensitive to even small changes
# so the above function might be overkill
# We may want to start really small, like one or two values at a time

# Another idea - classify different changes as different types of loss
# like lapse, fog, etc
# where one is changing weights, one is adding dropout layers, etc.