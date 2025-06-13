from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import process_2d
import process_svo2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', video_path=None, info_content=None)

@app.route('/upload', methods=['POST'])
def upload():
    video = request.files['video']
    filename = secure_filename(video.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'processed_' + filename)

    video.save(input_path)
    ext = os.path.splitext(filename)[1].lower()

    if ext == '.svo2':
        process_svo2.process(input_path, output_path)
        with open(output_path + '.txt', 'r') as f:
            info_content = f.read()
        return render_template('index.html', video_path=None, info_content=info_content)

    elif ext in ['.mp4', '.avi', '.mov']:
        process_2d.process(input_path, output_path)
        return render_template('index.html', video_path=output_path, info_content=None)

    else:
        return "<h3>Desteklenmeyen video formatı</h3>"

if __name__ == '__main__':
    app.run(debug=True)
