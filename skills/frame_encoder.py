import cv2
import base64


def encode_frame(frame):

    _, buffer = cv2.imencode(".jpg", frame)

    return base64.b64encode(buffer).decode()