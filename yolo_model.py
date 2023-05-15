import torch
import torchvision
import numpy
import cv2
import os
from time import perf_counter
from ultralytics import YOLO
from IPython.display import display, Image
from IPython import display

import supervision as sv
from bytetracker import BYTETracker
from supervision.draw.color import ColorPalette
from supervision.detection.core import Detections
from supervision import get_video_frames_generator
from supervision import BoxAnnotator
from util import YamlParser
HOME = os.getcwd()
print(HOME)

SAVE_VIDEO = True
TRACKER = 'bytetrack'

#  _____________________________________________________________
# | To Do:                                                      |
# | Optimise:                                                   |
# | Try upgrading to latest versions of pytorch and torchvision |
#  -------------------------------------------------------------

class ObjectDetection:

    def __init__(self, capture_index):

        self.capture_index = capture_index
        self.generator = get_video_frames_generator(capture_index)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = BoxAnnotator(color=(0, 255, 0), thickness=3, text_thickness=3, text_scale=1.5,)
        # dummy code color=(0, 255, 0), thickness=3, text_thickness=3, text_scale=1.5
        # use ColourPalette()

        # Tracker
        if TRACKER == 'bytetrack':
            tracker_config = 'C:/Users/Owner/PycharmProjects/thesis/venv/Lib/site-packages/bytetracker/config/byte_track.yaml'
            cfg = YamlParser()
            cfg.merge_from_file(tracker_config)

            self.tracker = BYTETracker(
                track_thresh=0.45,
                track_buffer=25,
                match_thresh=0.8,
                frame_rate=30,
        )

    def load_model(self):

        model = YOLO('yolov8s.pt')  # load a pretrained YOLOv8n model
        model.fuse()
        return model

    def predict(self, frame):
        results = self.predict(frame)
        print(results)
        return results

    def draw_results(self, results, frame):

        xyxys = []
        confidences = []
        class_ids = []
        boxes = []

        # Extract detections for person class
        for result in results:
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if class_id in [0, 32]:
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
                boxes.append(result.boxes)

                # Setup detections for visualization
                detections = sv.Detections(
                xyxy=results.xyxy.cpu().numpy(),
                confidence=result.conf.cpu().numpy(),
                class_id=result.boxes.cls.cpu().numpy().astype(int),
                )

            # Format custom labels
            self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections]

        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)

        return frame, boxes

    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if SAVE_VIDEO:
            outputvid = cv2.VideoWriter('result_tracking.mp4', fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=8, frameSize=(1280, 720))

        # setup tracker
        tracker = self.tracker

        # if tracker is using model then warmup
        if hasattr(tracker, 'model'):
            if hasattr(tracker.model, 'warmup'):
                tracker.model.warmup()

        outputs = [None]
        curr_frames, prev_frames = None, None

        while True:
            start_time = perf_counter()
            ret, frame = cap.read()

            assert ret
            results = self.predict(frame)

            frame, _ = self.draw_results(frame, results) # might have to swap these two args around

            # camera motion compensation
            if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'camera_update'):
                if prev_frames is not None and curr_frames is not None:
                    tracker.tracker.camera_update(prev_frames, curr_frames)

            for result in results:
                outputs[0] = tracker.update(result, frame)
                for i, (output) in enumerate(outputs[0]):
                    bbox = output[0:4]
                    tracked_id = output[4]
                    # class = output[5]
                    # conf = output[6]
                    top_left = (int(bbox[-2] - 100), int(bbox[1]))
                    cv2.putText(frame, f"ID : {tracked_id}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, int((0, 255, 0)), 3)

            end_time = perf_counter()
            fps = 1 / numpy.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('YOLOv8 Detection', frame)

            if SAVE_VIDEO:
                outputvid.write(frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        if SAVE_VIDEO:
            outputvid.release()
        cap.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index="C:/Users/Owner/PycharmProjects/football_player_tracking/Data/sligo_v_leitrim_AdobeExpress.mp4")
detector()