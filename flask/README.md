# Flask Audio Processing Application

This is a Flask application that allows users to easily demo senseflosser without using CLI after downloading the repo and installing dependencies.

## Features

- Upload audio files
- Process audio files using different models
- Play processed audiofile

## Installation

1. Follow directions in the primary repo [README](https://github.com/hyve9/senseflosser/tree/main)

## Usage

You can start the flask app in a couple of ways. Assuming you are in the base directory of the repo on your computer:

```
cd flask
flask run
```

Then, open a web browser and navigate to `http://localhost:5000` or `http://127.0.0.1:5000`.

Notes: 
* Model selection is based on what model files (`.h5`) are in `senseflosser/models`.
* Max duration is determined once a file is chosen and is not limited by the timeframe used to train the model (i.e. `15s_audio_autoencoder.h5` can process more than 15 seconds of audio)

## Endpoints

- `GET /`: Render the main page of the application.
- `POST /`: Process the uploaded audio file and return the result.
- `GET /output/<filename>`: Serve the processed audio file.
- `POST /upload`: Upload an audio file and return the maximum duration.