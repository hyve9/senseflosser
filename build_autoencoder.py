import sys
import os
import argparse
import logging
import tensorflow as tf
from pathlib import Path
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from autoencoder.model import (preprocess_input, 
                                     load_data, 
                                     build_model,
                                     SAMPLE_RATE,
                                     HOP_LEN,
                                     WINDOW_LEN,
                                     EPOCHS,
                                     BATCH_SIZE)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing the audio files')
    parser.add_argument('--duration', type=int, default=30, help='Duration of audio (in seconds) to use for training')
    parser.add_argument('--var-input', action='store_true', help='Use variable input size for model')
    parser.add_argument('--percentage', type=float, help='Percentage of dataset to use')
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
    duration = args.duration
    var_input = args.var_input
    percentage = args.percentage if args.percentage else 0.6
    logging.debug(f'Duration set to {duration}')
    if duration < 1:
        logging.error('Duration must be at least 1 second.')
        sys.exit(1)

    # Model params
    sequence_length = duration * SAMPLE_RATE - ((duration * SAMPLE_RATE) % WINDOW_LEN)
    freq_bins = WINDOW_LEN // 2 + 1
    windows = ((sequence_length - WINDOW_LEN) // HOP_LEN) + 1

    # Test the preprocess function
    if loglevel == 'debug':
        # Test example input
        preprocess_input(sequence_length, windows, freq_bins)
    
    # Load audio and split into datasets
    # Not using test atm
    train, val, _ = load_data(data_dir, sequence_length, windows, freq_bins, percentage)
    
    # Build the autoencoder
    autoencoder = build_model(windows, freq_bins, var_input=var_input)

    # look at data to make sure we aren't crazy
    if loglevel == 'debug':
        for x, y in train.take(1):
            logging.debug(f'Training x shape: {x.shape}')
            logging.debug(f'Training y shape: {y.shape}')
    
    # look at the model
    logging.debug(autoencoder.summary())

    # Callbacks
    ckpt_folder = Path('./checkpoints')
    if ckpt_folder.exists():
        latest = tf.train.latest_checkpoint(ckpt_folder)
        autoencoder.load_weights(latest)
    os.makedirs(ckpt_folder, exist_ok=True)
    ckpt_name = f'b{BATCH_SIZE}_e{EPOCHS}_d{duration}s_audio_autoencoder_best.ckpt'
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
        )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=5, 
        min_lr=0.001
        )
    model_ckpt = ModelCheckpoint(
        filepath=ckpt_folder.joinpath(ckpt_name),
        save_weights_only=True, 
        save_best_only=True, 
        monitor='val_loss', 
        mode='min', 
        verbose=1
        )

    # Train the model
    history = autoencoder.fit(x=train, epochs=EPOCHS, validation_data=val, callbacks=[early_stop, reduce_lr, model_ckpt], verbose=1)

    # Save model
    autoencoder.save(f'models/{duration}s_audio_autoencoder.h5')