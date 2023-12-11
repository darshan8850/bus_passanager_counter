from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
import face_detection
import numpy as np
import base64

app = Flask(__name__)

# Configuration for SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///faces.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Frame(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    frame_data = db.Column(db.LargeBinary)  # Use LargeBinary to store binary data
    count_of_people = db.Column(db.Integer)
    
def create_database():
    db.create_all()


def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

def detect_faces_and_save(frame):
    detector = face_detection.build_detector("DSFDDetector", confidence_threshold=0.5, nms_iou_threshold=0.3)
    det_raw = detector.detect(frame[:, :, ::-1])
    dets = det_raw[:, :4]
    draw_faces(frame, dets)

    count_of_people = len(dets)


    frame_data_encoded = base64.b64encode(cv2.imencode('.jpg', frame)[1].tobytes())
    new_frame = Frame(frame_data=frame_data_encoded, count_of_people=count_of_people)
    db.session.add(new_frame)
    db.session.commit()

    return frame, count_of_people

@app.route('/video_feed', methods=['POST'])
def video_feed():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file in the request'})

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({'error': 'Empty video file'})

    video_bytes = video_file.read()
    video_np = np.frombuffer(video_bytes, dtype=np.uint8)
    vidcap = cv2.imdecode(video_np, cv2.IMREAD_UNCHANGED)

    if vidcap is None:
        return jsonify({'error': 'Error decoding video file'})

    create_database()

    success, image = vidcap.read()

    while success:

        detect_faces_and_save(image)

        success, image = vidcap.read()

    return jsonify({'result': 'Video parsed'})

@app.route('/get_frames', methods=['GET'])
def get_frames():
    frames = Frame.query.all()
    frames_data = []

    for frame in frames:
        frames_data.append({
            'frame_data': frame.frame_data.decode('latin1'),
            'count_of_people': frame.count_of_people
        })
    
    db.session.query(Frame).delete()
    db.session.commit()
    
    return jsonify(frames_data)

if __name__ == '__main__':
    app.run(debug=True)
