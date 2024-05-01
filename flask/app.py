import sys
import argparse
from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import librosa

# Really hacky - let's fix this later
sys.path.append(str(Path(__file__).resolve().parent.parent))
from senseflosser.utils import run_senseflosser

app = Flask(__name__)

def get_audio_duration(input_file):
    y, sr = librosa.load(input_file)
    duration = librosa.get_duration(y=y, sr=sr)
    return duration

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_file = request.files['input_file']
        upload_dir = Path('uploads')
        upload_dir.mkdir(exist_ok=True)
        filename = Path(upload_dir, secure_filename(input_file.filename))  
        input_file.save(filename)
        max_duration = get_audio_duration(filename)

        model_name = request.form['model_name']
        magnitude = request.form['magnitude']
        duration = request.form['duration']
        action = request.form['action']

        # Convert duration and magnitude to int and float
        duration = int(duration)
        magnitude = [float(magnitude)]

        model_file = Path('../models', model_name)
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)

        # Call the function directly with parameters
        output_files = run_senseflosser(model_file, magnitude, action, filename, output_dir, duration, titrate=False, save_model=False)

        # Generate URLs for the output files
        output_file_urls = [url_for('serve_output', filename=file.name) for file in output_files]
        
        return jsonify({'output_file_urls': output_file_urls})
    else:
        model_dir = Path('../models')
        model_files = [f for f in model_dir.iterdir() if f.is_file()]
        max_duration = 30  # Default max duration
        return render_template('index.html', max_duration=max_duration, model_files=model_files)


@app.route('/output/<filename>')
def serve_output(filename):
    print('Serving file from:', Path('output', filename))
    return send_from_directory(Path('output'), filename)

@app.route('/upload', methods=['POST'])
def upload():
    input_file = request.files['file']
    upload_dir = Path('uploads')
    upload_dir.mkdir(exist_ok=True)  # Ensure the upload directory exists
    filename = secure_filename(input_file.filename)
    full_path = upload_dir / filename
    print("Saving file to", full_path)
    input_file.save(full_path)
    max_duration = get_audio_duration(full_path)
    file_url = url_for('serve_upload', filename=filename, _external=True)
    print(file_url)
    return jsonify({'max_duration': max_duration, 'file_url': file_url})

@app.route('/uploads/<filename>')
def serve_upload(filename):
    print("Serving file from:", Path('uploads', filename))
    return send_from_directory(Path('uploads'), filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Flask app')
    parser.add_argument('--port', type=int, default=5000, help='Port number to run the app on') 
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address to run the app on')
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=True)
