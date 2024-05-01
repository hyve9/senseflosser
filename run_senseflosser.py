import sys
import argparse
import logging
from pathlib import Path
from senseflosser.utils import run_senseflosser

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='models/5s_audio_autoencoder.h5', help='Model file to load')
    parser.add_argument('--magnitude', type=float, default=0.01, help='Magnitude of noise to introduce')
    parser.add_argument('--titrate', action='store_true', help='Titrate noise magnitude')
    parser.add_argument('--duration', type=int, help='Duration of audio in seconds')
    parser.add_argument('--action', type=str, default='fog', help='Action to perform (currently fog or lapse)')
    parser.add_argument('--input', type=str, help='Input file to process')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory (default: ./output)')
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
    input_file = Path(args.input)
    model_file = Path(args.model)
    output_dir = Path(args.output_dir)

    run_senseflosser(model_file, 
                     magnitude, 
                     action, 
                     input_file, 
                     output_dir, 
                     duration,
                     args.titrate,
                     args.save_model)


    
