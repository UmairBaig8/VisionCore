import cv2


def sample_frames(video_path, interval=0.5):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    step = int(fps * interval)

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_idx % step == 0:

            yield frame_idx / fps, frame

        frame_idx += 1

    cap.release()