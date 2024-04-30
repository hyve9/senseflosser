from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import subprocess
import argparse

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        audio_file = request.files['audio']
        model_name = request.form['model_name']
        magnitude = request.form['magnitude']

        filename = os.path.join(os.getcwd(), '..', 'uploads', secure_filename(audio_file.filename))
        audio_file.save(filename)
        
        model_filename = os.path.join('..', 'models', model_name)

        result = subprocess.run([
            'python', 
            '../run_senseflosser.py', 
            '--model-file', model_filename, 
            '--magnitude', magnitude,
            '--input', filename,
            '--output-dir', '../output'],
            capture_output=True,
            text=True)

        print('return code:', result.returncode)
        print('stdout:', result.stdout)
        print('stderr:', result.stderr)
        
        base_filename = os.path.basename(filename)
        output_filename = os.path.join(os.getcwd(), '..', 'output', base_filename.rsplit('.', 1)[0] + '_normal.wav')
        output_file_rel_path = os.path.relpath(output_filename, start=os.getcwd())
        output_file_url_path = output_file_rel_path.replace('\\', '/') # for Windows but idk if it's necessary
        print('output_file_rel_path:', output_file_rel_path)
        output_file_url = request.url_root + output_file_url_path
        print('output_file_url:', output_file_url)
        return jsonify({'result': result.stdout, 'output_file_url': output_file_url})
    else:
        model_dir = os.path.join(os.getcwd(), '..', 'models')
        model_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
        return render_template('index.html', model_files=model_files)


@app.route('/output/<filename>')
def serve_output(filename):
    return send_from_directory(os.path.join(os.getcwd(), '..', 'output'), filename)

if __name__ == '__main__':
    app.run(debug=True)