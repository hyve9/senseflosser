import sys
import argparse
import logging
import librosa
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from senseflosser import degenerate_model, preprocess_input

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

    breakpoint()
    y, sr = preprocess_input(y, orig_sr, orig_model)
    normal_output = orig_model.predict(y)

    # Introduce degradation
    degraded_model = degenerate_model(orig_model, magnitude)
    degraded_output = degraded_model.predict(y)

    # Write waveforms
    output_file_prefix = input.stem
    y = librosa.resample(y, orig_sr=sr, target_sr=orig_sr)
    librosa.output.write_wav(f'{output_file_prefix}_normal.wav', normal_output, sr)
    librosa.output.write_wav(f'{output_file_prefix}_degraded.wav', degraded_output, sr)

    # Save degraded model
    output_model_prefix = model_file.stem
    degraded_model.save(f'{output_model_prefix}_degraded.h5')

    