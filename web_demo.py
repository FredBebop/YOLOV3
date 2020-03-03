#! /usr/bin/env python
# coding=utf-8

import cv2
import time
import numpy as np
from flask import Response,Flask,render_template

import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
import threading


outputFrame = None
lock = threading.Lock()
vid = cv2.VideoCapture(0)

app = Flask(__name__)

def detect_yolov3(vid,model,input_size):

    # model.summary()
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("No image!")
    frame_size = frame.shape[:2]
    image_data = utils.image_preprocess(np.copy(frame), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)

    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = utils.preprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    image = utils.draw_bbox(frame, bboxes)
    with lock:
        outputFrame =image
        # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        # result = cv2.cvtColor(outputFrame, cv2.COLOR_RGB2BGR)
        # cv2.imshow("result", result)


#--------------------------falsk调用OpenCV和YOLOv3------------------------------------

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

def detect():
    global vid, outputFrame, lock
    num_classes = 80
    input_size = 416

    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv3(input_layer)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, "./yolov3.weights")
    while True:
        detect_yolov3(vid,model,input_size)


def generate():
    global outputFrame, lock

    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag,encodedImage) = cv2.imencode(".jpg",outputFrame)

            if not flag:
                continue
        yield(b"--frame\r\n" b"Content-Type:image/jpeg\r\n\r\n"+bytearray(encodedImage)+b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(generate(),mimetype='multipart/x-mixed-replace;boundary=frame')




if __name__ == "__main__":
    t = threading.Thread(target=detect)
    t.daemon = True
    t.start()
    app.run(debug=True,threaded=True,use_reloader=False)

#release视频流
vid.stop()



