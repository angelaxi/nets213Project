from flask import Flask, render_template, Response, jsonify
from camera import VideoCamera
import cv2
from bs4 import BeautifulSoup

app = Flask(__name__)

video_stream = VideoCamera()

@app.route('/')
def index():
    update_label()
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def update_label():
    with open('templates/index.html') as html_file:
        soup = BeautifulSoup(html_file.read(), features='html.parser')
        tag = soup.find(id="label")
        if True:
            tag.string.replace_with('Mask: On')
        else:
            tag.string.replace_with('Mask: Off')
        new_text = soup.prettify()
    with open('templates/index.html', mode='w') as new_html_file:
        new_html_file.write(new_text)


@app.route('/video_feed')
def video_feed():
    return Response(gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True,port="5000")