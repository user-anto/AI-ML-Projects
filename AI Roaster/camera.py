import os

import cv2


def open_camera(camera_index: int):
    # On Windows, DirectShow is often more stable than the default backend.
    if os.name == "nt":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(camera_index)
