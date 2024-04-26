import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import librosa
from build_autoencoder import SAMPLE_RATE
from keras.layers import Dropout, LSTM
from keras.models import Model


def window_audio(y, window_size):
    # Split audio into windows
    windows = []
    for i in range(0, len(y), window_size):
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

def fog(model, magnitude):
    for i, layer in enumerate(model.layers):
        # Only run fog on LSTM layers; may do others later
        if isinstance(layer, LSTM):
            if i % 1 == 0:
                weights = layer.get_weights()
                new_weights = []
                for weight_matrix in weights:
                    # Introduce random noise or zero out weights
                    shape = weight_matrix.shape
                    noise = np.random.normal(loc=0.0, scale=magnitude, size=shape)
                    new_weight_matrix = weight_matrix + noise
                    new_weights.append(new_weight_matrix)
                model.layers[i].set_weights(new_weights)
    return model

def lapse(model, magnitude):
    new_model_layers = []
    input_layer = model.input
    x = input_layer
    
    for i, layer in enumerate(model.layers):
        # Clone the layer from the original model configuration
        cloned_layer = layer.__class__.from_config(layer.get_config())
        x = cloned_layer(x)
        cloned_layer.set_weights(layer.get_weights())
        new_model_layers.append(cloned_layer)
        
        if isinstance(layer, LSTM):
            if i % 1 == 0:
                dropout_layer = Dropout(magnitude)
                x = dropout_layer(x)
                new_model_layers.append(dropout_layer)
                
    # Create new model based on the functional API
    new_model = Model(inputs=input_layer, outputs=x)
    new_model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    return new_model

def floss_model(model, magnitude=0.1, action='fog'):
    if 'LSTM' not in model.summary():
        logging.warn('Model does not contain any LSTM layers; model will remain unmodified')
        return model
    if action == 'fog':
        model = fog(model, magnitude)
    elif action == 'lapse':
        model = lapse(model, magnitude)
    else:
        logging.warn('Invalid action; model will remain unmodified')
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