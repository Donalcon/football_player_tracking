import sys
import os
HOME = os.getcwd()
print(HOME)
sys.path.append('C:/Users/Owner/PycharmProjects/football_player_tracking/ByteTrack')
import yaml

import cv2
import torch
import numpy as np
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker





from roboflow import Roboflow
rf = Roboflow(api_key="hfGn1GDEH5TgyhZuTgiG")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
dataset = project.version(1).download("yolov8")

yolo_data = '.Yolo/model_data/data.yaml'
with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])



# YOLO object detection function
def detect_objects(image, model):
    # Preprocess the image
    img = torch.from_numpy(image).float().div(255.0).unsqueeze(0)
    img = img.permute(0, 3, 1, 2)

    # Perform object detection
    with torch.no_grad():
        detections = model(img)[0]
        detections = non_max_suppression(detections, conf_thres, iou_thres)

    return detections

# YOLO model setup
model_path = 'C:/Users/Owner/PycharmProjects/football_player_tracking/Yolo/weights/best.pt'
conf_thres = 0.5
iou_thres = 0.5

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', = 'C:/Users/Owner/PycharmProjects/football_player_tracking/Yolo/weights/best.pt', force_reload=True)

# ByteTrack setup
tracker = BYTETracker()

# Video capture setup
video_path = 'path/to/video/file.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection using YOLO
    detections = detect_objects(frame, yolo_model)

    # Object tracking using ByteTrack
    tracked_objects = tracker.update(detections)

    # Visualize the results
    for obj in tracked_objects:
        x1, y1, x2, y2, _, _, id = obj
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Detection and Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()