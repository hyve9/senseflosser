import sys
import argparse
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-file', type=str, default='models/audio_autoencoder.h5', help='Model file to load')
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
    model_file = Path(args.model_file)
    model = keras.models.load_model(model_file)