from typing import Generator
from Yolo.load_models import yoloV5m
from Yolo.detector import detectXl5

import matplotlib.pyplot as plt
import numpy as np

import cv2


def generate_frames(video_file: str) -> Generator[np.ndarray, None, None]:
    video = cv2.VideoCapture(video_file)

    while video.isOpened():
        success, frame = video.read()

        if not success:
            break

        yield frame

    video.release()


def plot_image(image: np.ndarray, size: int = 12) -> None:
    plt.figure(figsize=(size, size))
    plt.imshow(image[...,::-1])
    plt.show()
    plt.show(block=False)


SOURCE_VIDEO_PATH = 'C:/Users/Owner/PycharmProjects/football_player_tracking/Data/sligo_v_leitrim_AdobeExpress.mp4'

frame_iterator = iter(generate_frames(video_file=SOURCE_VIDEO_PATH))

frame = next(frame_iterator)
plot_image(frame, 16)


def detect_demo(path="./Data/gaa_image_data/1.JPG"):
    '''
    this functions is to apply detection on an image.

    Parameters
    ----------
    path : string
        the path of the directory of the processed image.
    '''

    modeli, ball_model = yoloV5m()
    img, res = detectXl5(modeli, path, show=True)
    print(f'bounding boxes: {res}')

    ball_img, ball_res = detectXl5(ball_model, path, show=True)
    print(ball_res)

    # show one player
    player = res[5]
    print(f'player with index 5: {player}')
    # cv2.imshow(img[player[1] - 30:player[3] + 30, player[0] - 30 :player[2] + 30, ::-1])
    plt.figure(figsize=(12, 8))
    # plt.imshow(img[player[1] - 30:player[3] + 30, player[0] - 30:player[2] + 30, :])
    plt.imshow(cv2.cvtColor(img[player[1] - 30:player[3] + 30, player[0] - 30:player[2] + 30, :], cv2.COLOR_BGR2RGB))
    plt.show()
    return img, res

detect_demo(path="./Data/gaa_image_data/1.JPG")
