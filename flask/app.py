import sys
from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path

# Really hacky - let's fix this later
sys.path.append(str(Path(__file__).resolve().parent.parent))
from senseflosser.utils import run_senseflosser

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_file = request.files['input_file']
        model_name = request.form['model_name']
        magnitude = request.form['magnitude']
        action = request.form['action']

        upload_dir = Path('uploads')
        upload_dir.mkdir(exist_ok=True)
        filepath = Path(upload_dir.joinpath(secure_filename(input_file.filename)))
        input_file.save(filepath)
        
        model_dir = Path('../models')
        model_file = model_dir.joinpath(model_name)
        output_dir = Path('../output')

        # Call the function directly with parameters
        output_files = run_senseflosser(model_file, magnitude, action, input_file, output_dir, duration=None, titrate=False, save_model=False)

        # Generate URLs for the output files
        output_file_urls = [url_for('serve_output', filename=file.name) for file in output_files]

        return jsonify(output_files=output_file_urls)
    else:
        return render_template('index.html', model_files=Path('models').glob('*.h5'), actions=['fog', 'lapse'])


@app.route('/output/<filename>')
def serve_output(filename):
    return send_from_directory(Path('../output'), filename)

if __name__ == '__main__':
    app.run(debug=True)