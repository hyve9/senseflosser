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
    parser.add_argument('--magnitude', type=float, default=0.01, help='Magnitude of noise to introduce')
    parser.add_argument('--titrate', action='store_true', help='Titrate noise magnitude')
    parser.add_argument('--duration', type=int, help='Duration of audio in seconds')
    parser.add_argument('--action', type=str, default='fog', help='Action to perform (currently fog or lapse)')
    parser.add_argument('--input', type=str, help='Input file to process')
    parser.add_argument('--save-model', action='store_true', help='Save flossed model')
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
    magnitude = [args.magnitude]
    if args.titrate:
        magnitude = [0.01, 0.05, 0.10, 0.20, 0.50]
    if args.titrate and args.magnitude:
        logging.warning('Titrate takes precedence over magnitude; ignoring magnitude if specified...')
    # Really need something more robust here
    duration = args.duration if args.duration else None
    action = args.action
    if action not in ['fog', 'lapse']:
        logging.error('Action must be either fog or lapse')
        sys.exit(1)
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

    # Get flossin'!
    flossed_outputs = dict()
    for i, m in enumerate(magnitude):
        flossed_model = floss_model(orig_model, magnitude[i], action)
        flossed_output = flossed_model.predict(y_proc)
        flossed_outputs[magnitude[i]] = postprocess_output(flossed_output, WINDOW_LEN, HOP_LEN, WTYPE)

    # Write waveforms
    work_folder = Path('./output')
    os.makedirs(work_folder, exist_ok=True)
    output_file_prefix = input.stem
    wavfile.write(work_folder.joinpath(f'{output_file_prefix}_normal.wav'), SAMPLE_RATE, normal_output)
    for m in flossed_outputs:
        wavfile.write(work_folder.joinpath(f'{output_file_prefix}_{action}_{m}.wav'), SAMPLE_RATE, flossed_outputs[m])

    # Save flossed model
    if args.save_model:
        if args.titrate:
            logging.error('Not saving titrated models; please specify a single magnitude.')
            sys.exit(0)
        model_folder = Path('./models')
        os.makedirs(model_folder, exist_ok=True)
        output_model_prefix = model_file.stem
        flossed_model.save(model_folder.joinpath(f'{output_model_prefix}_{action}_{magnitude[0]}.h5'))
    