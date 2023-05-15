import torch
import os
HOME = os.getcwd()
print(HOME)
#  __________________________________________________________________________________
# | Transfer this info to function in load_model attribute in object detection class |
# '----------------------------------------------------------------------------------'


#WEIGHTS_PATH = "./Yolo/weights/best.pt"

from ultralytics import YOLO

# custom model, can't run as have insufficient memory
# model = YOLO('yolov8s.pt')
# model.train(
#    model = 'yolov8s.pt',
#    data = 'C:/Users/Owner/PycharmProjects/football_player_tracking/football-players-detection-1/data.yaml',
#    epochs=100,
#    imgsz=640,
#    name='yolov8s_custom'
#)

# validate model, change args
# model.detect(
#    model = 'yolov8s.pt',
#    data = 'C:/Users/Owner/PycharmProjects/football_player_tracking/football-players-detection-1/data.yaml',
#    epochs=100,
#    imgsz=640,
#    name='yolov8s_custom'
#)

# use pre-trained model
model = YOLO('yolov8s.pt')
