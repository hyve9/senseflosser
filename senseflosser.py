import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging

def process_input(y, sr, model):
    new_sr = 22050
    y = librosa.utils.resample(y, sr, new_sr)
    hardcoded_shape = (44100, 1)
    y = y[:44100].reshape(hardcoded_shape)
    y = tf.convert_to_tensor(y, dtype=tf.float23)
    return y, new_sr

def degenerate_ltsm(layer, magnitude):
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

def degenerate_dense(layer):
    # Example: Manipulating Dense layer weights
    weights = layer.get_weights()
    new_weights = []
    for weight_matrix in weights:
        # do something here
        pass
    return layer

def degenerate_model(model, magnitude=0.1):
    if 'LSTM' not in model.summary():
        logging.warn('Model does not contain any LSTM layers; model will remain unmodified')
        return model
    for i in range(len(model.layers)):
        if isinstance(model.layers[i], keras.layers.LSTM):
            model.layers[i] = degenerate_ltsm(model.layers[i], magnitude)
        elif isinstance(model.layers[i], keras.layers.Dense):
            model.layers[i] = degenerate_dense(model.layers[i])
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