from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
import face_detection
import numpy as np
import base64
import os
import shutil
import asyncio
import threading

app = Flask(__name__)

# Configuration for SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///frame.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

detector = face_detection.build_detector("DSFDDetector", confidence_threshold=0.5, nms_iou_threshold=0.3)

class Frame(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    frame_data = db.Column(db.LargeBinary)  # Use LargeBinary to store binary data
    count_of_people = db.Column(db.Integer)

    def __init__(self,frame_data,count_of_people):
        self.frame_data=frame_data
        self.count_of_people=count_of_people


with app.app_context():
    db.create_all()

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

async def detect_faces_and_save(vidObj, media_folder):
    with app.app_context():
        # Get video properties
        fps = vidObj.get(cv2.CAP_PROP_FPS)
        total_frames = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps

        # Calculate frame sampling interval
        target_fps = 1  # One frame per second
        sampling_interval = int(fps / target_fps)

        success, image = vidObj.read()
        frame_counter = 0

        while success:
            if frame_counter % sampling_interval == 0:
                
                det_raw = detector.detect(image[:, :, ::-1])
                dets = det_raw[:, :4]
                draw_faces(image, dets)

                count_of_people = len(dets)

                frame_data_encoded = base64.b64encode(cv2.imencode('.jpg', image)[1].tobytes())
                new_frame = Frame(frame_data=frame_data_encoded, count_of_people=count_of_people)
                db.session.add(new_frame)
                db.session.commit()

            success, image = vidObj.read()
            frame_counter += 1

            if not success:
                vidObj.release()  # Release the video capture object before breaking out of the loop
                break

    # After the loop, release the video capture object and delete the file
    vidObj.release()
    shutil.rmtree(media_folder)

def process_upload_thread(vidObj, media_folder):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(detect_faces_and_save(vidObj, media_folder))
    loop.close()


@app.route('/video_feed', methods=['POST'])
def video_feed():

    media_folder = "media"
    if not os.path.exists(media_folder):
        os.makedirs(media_folder)

    if 'video' not in request.files:
        return jsonify({'error': 'No video file in the request'})

    video_file = request.files['video']
    video_path = os.path.join(media_folder, "uploaded_video.mp4")
    video_file.save(video_path)

    vidObj = cv2.VideoCapture(video_path)

    success, image = vidObj.read()
    
    det_raw = detector.detect(image[:, :, ::-1])
    dets = det_raw[:, :4]
    draw_faces(image, dets)

    count_of_people = len(dets)
    frame_data_encoded = base64.b64encode(cv2.imencode('.jpg', image)[1].tobytes())
    frame_data_encoded_str = frame_data_encoded.decode('latin1')
    
    threading.Thread(target=process_upload_thread, args=(vidObj, media_folder)).start()
    
    return jsonify({'frame': frame_data_encoded_str, 'count_of_people': count_of_people, "id":0})



@app.route('/')
def home():
    return jsonify(message='Welcome to Bus Passenger Counter!')

@app.route('/get_frames', methods=['GET'])
def get_frames():
    frame_id = request.args.get('id')
    frame = Frame.query.get(frame_id)

    if frame:
        frame_data = {
            'id': frame.id,
            'frame': frame.frame_data.decode('latin1'),
            'count_of_people': frame.count_of_people
        }
        return jsonify(frame_data)
    else:
        return jsonify({'error': 'Frame not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
