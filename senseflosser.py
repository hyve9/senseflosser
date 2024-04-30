import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import librosa
from keras.layers import Dropout, Conv2D, Conv2DTranspose
from keras.models import Model, Sequential

class DropoutAtInference(Dropout):
    # Adding dropout without training does not work
    # See: https://keras.io/api/layers/regularization_layers/dropout/ for details
    def call(self, inputs, training=None):
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

def lapse_old(model, magnitude):
    logging.debug('Entering ' + sys._getframe().f_code.co_name)
    new_model_layers = []
    input_layer = model.input
    x = input_layer
    
    for i, layer in enumerate(model.layers):
        # Clone the layer from the original model configuration
        cloned_layer = layer.__class__.from_config(layer.get_config())
        x = cloned_layer(x)
        cloned_layer.set_weights(layer.get_weights())
        new_model_layers.append(cloned_layer)
        
        if isinstance(layer, Conv2D) or isinstance(layer, Conv2DTranspose):
            dropout_layer = Dropout(magnitude)
            x = dropout_layer(x)
            new_model_layers.append(dropout_layer)
                
    # Create new model based on the functional API
    new_model = Model(inputs=input_layer, outputs=x)
    new_model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    return new_model

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


            
# more ideas

# something that tries to mimic how vinyl or tape might deteriorate
# slowed/reverbed (but how to translate that into latent space?)

# FWIW, I've been told that LTSM layers are quite sensitive to even small changes
# so the above function might be overkill
# We may want to start really small, like one or two values at a time

# Another idea - classify different changes as different types of loss
# like lapse, fog, etc
# where one is changing weights, one is adding dropout layers, etc.