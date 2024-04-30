# senseflosser

Takes an autoencoder keras model and adversarially attempts to deteriorate layers and neurons.

"senseflosser" is a [homophonic translation](https://en.wikipedia.org/wiki/Homophonic_translation) of Sinneslöschen, itself a non-idiomatic translation of "sense deletion", from the [Polybius mythos](https://en.wikipedia.org/wiki/Polybius_(urban_legend)).

### Setup: Create conda environment

```
conda create -f environment.yml -n senseflosser 
conda activate senseflosser
```

# Usage

```
usage: build_autoencoder.py [-h] --data-dir DATA_DIR [--duration DURATION] [--var-input] [--percentage PERCENTAGE]
                            [--log LOG]

optional arguments:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   Directory containing the audio files
  --duration DURATION   Duration of audio (in seconds) to use for training
  --var-input           Use variable input size for model
  --percentage PERCENTAGE
                        Percentage of dataset to use
  --log LOG             Logging level (choose from: critical, error, warn, info, debug)
```

```
usage: run_senseflosser.py [-h] [--model-file MODEL_FILE] [--magnitude MAGNITUDE] [--titrate] [--duration DURATION]
                           [--action ACTION] [--input INPUT] [--save-model] [--log LOG]

optional arguments:
  -h, --help            show this help message and exit
  --model-file MODEL_FILE
                        Model file to load
  --magnitude MAGNITUDE
                        Magnitude of noise to introduce
  --titrate             Titrate noise magnitude
  --duration DURATION   Duration of audio in seconds
  --action ACTION       Action to perform (currently fog or lapse)
  --input INPUT         Input file to process
  --save-model          Save flossed model
  --log LOG             Logging level (choose from: critical, error, warn, info, debug)
```

## Autoencoder

This repo contains two tools (which should probably be separated into different projects). The first simply builds an audio autoencoder. This should reproduce audio input. To build your own autoencoder, you can run, for example:

```
python build_autoencoder.py --data-dir data/your_dataset --percentage 0.4 --duration 10 --log debug --var-input
```

This will build an autoencoder trained on 10 second samples grabbed from 40% of your data, and will build the model in such a way that it supports inference (prediction) on variable length input. 
The model will be saved under `models/<duration>s_audio_autoencoder.h5`

### Data preprocessing

The autoencoder is designed to work with 22 kHz, 16-bit, single-channel audio `.wav` files. Any other file format or encoding will cause the model to fail, although I'm sure it can be designed to be more robust. Submit a PR if you have an idea!

If you aren't lucky enough to have data that is in that format, there are two scripts that can do this for you. One will iterate through your data directory and convert all files from <extension> to `.wav`, reduce to mono, and resample to 22050. To use:

```
./scripts/fma_wav_full_converter.sh data/your_data_dir mp3
```

If you just want to mix to single channel and resample (you already have wav files):

```
./scripts/fma_wav_resampler.sh data/your_data_dir
```

This project used the 

## Senseflosser

Once you've built your autoencoder (or maybe you brought your own), you can experiment with degrading some layers by running senseflosser:

```
python run_senseflosser.py --input data/fma_small/006/006329.wav --model models/15s_audio_autoencoder.h5 --action lapse --magnitude 0.05 --duration 30
```

This takes an input audio file, a pre-trained autoencoder, an action (here either "fog" or "lapse"), a magnitude (here 0.05), and a duration for the audio. Note that you can specify the duration to be longer than the inputs that the model was trained on, only if the autoencoder was trained with the `--var-input` option. 

Based on experimentation, good values for magnitude are between 0.02 and 0.5. Values higher than 0.5 severely degrade the audio (which may be what you want!)

If you want to save your degraded model, add the `--save-model` option.
## Acknowledgments

This project was built for the Spring '24 Deep Learning for Media class at NYU. 

- **FMA Dataset**: The autoencoders included in this repo were trained on (preprocessed) data from the [FMA Dataset](https://github.com/mdeff/fma), and the two scripts mentioned above expect a similar directory structure. The FMA dataset is licensed under the MIT License.

