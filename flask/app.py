from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import subprocess
import argparse
import librosa

def get_audio_duration(audio_file):
    y, sr = librosa.load(audio_file)
    duration = librosa.get_duration(y=y, sr=sr)
    return duration

app = Flask(__name__)

script_dir = os.path.dirname(os.path.realpath(__file__))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        audio_file = request.files['audio']
        filename = os.path.join(script_dir, 'uploads', secure_filename(audio_file.filename))  # Removed '..'
        audio_file.save(filename)
        max_duration = get_audio_duration(filename)

        model_name = request.form['model_name']
        magnitude = request.form['magnitude']
        # titrate = request.form['titrate']
        duration = request.form['duration']
        action = request.form['action']

        model_filename = os.path.join('..', 'models', model_name)

        # result = subprocess.run([
        #     'python', 
        #     '../run_senseflosser.py', 
        #     '--model', model_filename, 
        #     '--magnitude', magnitude,
        #     #'--titrate', # not sure yet
        #     '--duration', duration,
        #     '--action', action,
        #     '--input', filename,
        #     '--output-dir', '../output'],
        #     capture_output=True,
        #     text=True)
        result = subprocess.run([
            'python', 
            os.path.join(script_dir, '..', 'run_senseflosser.py'),  # Use an absolute path
            '--model', model_filename, 
            '--magnitude', magnitude,
            #'--titrate', # not sure yet
            '--duration', duration,
            '--action', action,
            '--input', filename,
            '--output-dir', os.path.join(script_dir, '..', 'output')],  # Use an absolute path
            capture_output=True,
            text=True)


        print('return code:', result.returncode)
        print('stdout:', result.stdout)
        print('stderr:', result.stderr)
        
        base_filename = os.path.basename(filename)
        output_filename = os.path.join(script_dir, '..', 'output', base_filename.rsplit('.', 1)[0] + '_normal.wav')
        output_file_url_path = os.path.relpath(output_filename, start=script_dir)
        # output_file_url_path = output_file_rel_path.replace('\\', '/') # for Windows but idk if it's necessary
        print('output_file_rel_path:', output_file_url_path)
        output_file_url = request.url_root + output_file_url_path
        print('output_file_url:', output_file_url)
        return jsonify({'result': result.stdout, 'output_file_url': output_file_url})
    else:
        model_dir = os.path.join(script_dir, '..', 'models')
        model_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
        max_duration = 30  # Default max duration
        return render_template('index.html', max_duration=max_duration, model_files=model_files)


@app.route('/output/<filename>')
def serve_output(filename):
    return send_from_directory(os.path.join(script_dir, '..', 'output'), filename)

@app.route('/upload', methods=['POST'])
def upload():
    audio_file = request.files['file']
    upload_dir = os.path.join(script_dir, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)  # Ensure the upload directory exists
    filename = os.path.join(upload_dir, secure_filename(audio_file.filename))
    audio_file.save(filename)
    max_duration = get_audio_duration(filename)
    file_url = request.url_root + 'uploads/' + filename
    return jsonify({'max_duration': max_duration, 'file_url': file_url})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Flask app')
    parser.add_argument('--port', type=int, default=5000, help='Port number to run the app on') 
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address to run the app on')
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=True)