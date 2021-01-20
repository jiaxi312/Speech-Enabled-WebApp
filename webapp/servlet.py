import os
import davenet_demo.models as models

from flask import Flask, render_template, request, redirect, url_for, Response
from match_mapper import MatchMapper, AUDIO_MODEL_PATH, IMAGE_MODEL_PATH


audio_model, image_model = models.DAVEnet_model_loader(AUDIO_MODEL_PATH, IMAGE_MODEL_PATH)
mm = MatchMapper(audio_model, image_model)

VIDEO_FOLDER = os.path.join('static', 'files')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = VIDEO_FOLDER


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo')
def demo():
    video_file = os.path.join(app.config['UPLOAD_FOLDER'], 'output.mp4')
    return render_template('demo.html', video=video_file)


@app.route('/audio-data', methods=['POST'])
def receive_audio():
    if 'audio' not in request.files:
        return redirect(url_for('index'))
    upload_file = request.files['audio']
    if upload_file.filename != '':
        print('start')
        upload_file.save('./webapp/static/files/' + upload_file.filename)
        image_name = request.form.get('image_name')
        mm.demo_with_animation('./webapp/static/img/' + image_name, './webapp/static/files/' + upload_file.filename)
        print('finish')
    return redirect(url_for('demo'))


if __name__ == "__main__":
    app.run(debug=True)
