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
from senseflosser import floss_model, preprocess_input

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-file', type=str, default='models/2s_audio_autoencoder.h5', help='Model file to load')
    parser.add_argument('--magnitude', type=float, default=0.1, help='Magnitude of noise to introduce')
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
    input = Path(args.input)
    model_file = Path(args.model_file)
    orig_model = keras.models.load_model(model_file)

    # Obtain normal output
    y, orig_sr = librosa.load(input)

    y_proc, sr = preprocess_input(y, orig_sr, orig_model)
    output_windows = []
    for window in y_proc:
        output = orig_model.predict(window)
        output = np.squeeze(output, axis=(0,2))
        output_windows.append(output)
    normal_output = np.concatenate(output_windows)
    breakpoint()

    # Introduce degradation
    # flossed_model = floss_model(orig_model, magnitude)
    # flossed_output = flossed_model.predict(y)

    # Write waveforms
    work_folder = Path('./output')
    os.makedirs(work_folder, exist_ok=True)
    output_file_prefix = input.stem
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=orig_sr)
    wavfile.write(work_folder.joinpath(f'{output_file_prefix}_normal.wav'), orig_sr, normal_output)
    #wavfile.write(work_folder.joinpath(f'{output_file_prefix}_flossed.wav'), orig_sr, flossed_output)

    # Save flossed model
    # output_model_prefix = model_file.stem
    # flossed_model.save(f'{output_model_prefix}_flossed.h5')

    