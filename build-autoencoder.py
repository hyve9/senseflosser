import os
import sys
import argparse
import logging
import librosa
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Constants
SAMPLE_RATE = 22050
DURATION = 30
N_FEATURES = 1
EPOCHS = 50
BATCH_SIZE = 10
UNITS = 128

# Model params
timesteps = SAMPLE_RATE * DURATION
input_dim = N_FEATURES

def load_data(data_dir, sample_rate, duration):
    audio_data = []
    for root, dirs, files in os.walk(data_dir):
            for f in files:
                full_path = Path(root, f)
                if full_path.suffix == '.wav':
                    audio, sr = librosa.load(full_path, sr=sample_rate, duration=duration)
                    if sr != sample_rate:
                        audio = librosa.resample(audio, sr, sample_rate)
                    if len(audio) > sample_rate * duration:
                        audio = audio[:sample_rate * duration]
                    if len(audio) < sample_rate * duration:
                        audio = np.pad(audio, (0, sample_rate * duration - len(audio)))
                    audio_data.append(audio)
    return np.array(audio_data)

def build_model(timesteps, input_dim):
    input_layer = Input(shape=(timesteps, input_dim))
    # Long Short-Term Memory (LSTM), a type of recurrent neural network (https://keras.io/api/layers/recurrent_layers/lstm/)
    # Using 128 units in the LSTM layer is a common choice for audio data
    encoder = LSTM(UNITS)(input_layer)
    # Repeat the input timesteps times
    repeated = RepeatVector(timesteps)(encoder)
    # return_sequences=True means that the LSTM layer will return the full sequence of outputs for each input
    # This is what we want, audio in == audio out, nothing fancy
    decoder = LSTM(input_dim, return_sequences=True)(repeated)

    autoencoder = Model(input_layer, decoder)
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

    # Load waveforms
    audio_data = load_data(data_dir, SAMPLE_RATE, DURATION)

    # Build autoencoder
    autoencoder = build_model(timesteps, input_dim)

    # Train the model
    autoencoder.fit(audio_data, audio_data, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

    # Save model
    autoencoder.save('models/audio_autoencoder.h5')
