import cv2


def open_video(path):

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")

    return cap