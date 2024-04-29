import sys
import os
import argparse
import logging
import librosa
from scipy.io import wavfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from senseflosser import floss_model, postprocess_output
from build_autoencoder import (preprocess_input,
                                 SAMPLE_RATE,
                                 HOP_LEN,
                                 WINDOW_LEN,
                                 WTYPE
                                 )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-file', type=str, default='models/5s_audio_autoencoder.h5', help='Model file to load')
    parser.add_argument('--magnitude', type=float, default=0.1, help='Magnitude of noise to introduce')
    parser.add_argument('--duration', type=int, help='Duration of audio in seconds')
    parser.add_argument('--input', type=str, help='Input file to process')
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
    magnitude = args.magnitude
    # Really need something more robust here
    duration = args.duration if args.duration else None
    input = Path(args.input)
    model_file = Path(args.model_file)
    orig_model = keras.models.load_model(model_file)

    # Obtain normal output
    y, sr = librosa.load(input, mono=True)

    # Model params
    if duration is None:
        try:
            duration = int(model_file.stem.split('s')[0])
        except ValueError: # If model file name doesn't have a duration in it
            logging.error('Duration must be specified if model file name does not contain duration')
            sys.exit(1)
    sequence_length = duration * SAMPLE_RATE - ((duration * SAMPLE_RATE) % WINDOW_LEN)
    freq_bins = WINDOW_LEN // 2 + 1
    windows = ((sequence_length - WINDOW_LEN) // HOP_LEN) + 1
    
    # Preprocess input
    y_proc = preprocess_input(sequence_length, windows, freq_bins, y, sr)

    # Predict
    S_normal_output = orig_model.predict(y_proc)
    normal_output = postprocess_output(S_normal_output, WINDOW_LEN, HOP_LEN, WTYPE)

    # Introduce degradation
    # flossed_model = floss_model(orig_model, magnitude)
    # flossed_output = flossed_model.predict(y)

    # Write waveforms
    work_folder = Path('./output')
    os.makedirs(work_folder, exist_ok=True)
    output_file_prefix = input.stem
    if SAMPLE_RATE != sr:
        # Resample back to original sample rate
        y = librosa.resample(y, orig_sr=SAMPLE_RATE, target_sr=sr)
    wavfile.write(work_folder.joinpath(f'{output_file_prefix}_normal.wav'), sr, normal_output)
    #wavfile.write(work_folder.joinpath(f'{output_file_prefix}_flossed.wav'), sr, flossed_output)

    # Save flossed model
    # model_folder = Path('./models')
    # os.makedirs(model_folder, exist_ok=True)
    # output_model_prefix = model_file.stem
    # flossed_model.save(model_folder.joinpath(f'{output_model_prefix}_flossed.h5'))

    