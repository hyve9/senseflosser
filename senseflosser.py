import numpy as np
from tensorflow import keras

def degenerate(model):
    # Example: Manipulating LSTM layer weights
    for layer in model.layers:
        if isinstance(layer, keras.layers.LSTM):
            weights = layer.get_weights()
            new_weights = []
            for weight_matrix in weights:
                # Introduce random noise or zero out weights
                shape = weight_matrix.shape
                noise = np.random.normal(loc=0.0, scale=0.1, size=shape)
                new_weight_matrix = weight_matrix * (1 + noise)
                new_weights.append(new_weight_matrix)
            layer.set_weights(new_weights)

    return model


# more ideas

# something that tries to mimic how vinyl or tape might deteriorate
# slowed/reverbed (but how to translate that into latent space?)

# FWIW, I've been told that LTSM layers are quite sensitive to even small changes
# so the above function might be overkill
# We may want to start really small, like one or two values at a time