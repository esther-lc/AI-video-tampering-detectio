import cv2
import numpy as np
import os

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        frames.append(smooth)

    cap.release()

    frames = np.array(frames)
    total_frames = frames.shape[0]
    duration = total_frames / fps if fps > 0 else 0

    base_dir = "feature_extraction/preprocessed"
    os.makedirs(base_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(base_dir, f"{video_name}.npy")
    np.save(save_path, frames)

    return frames, total_frames, fps, duration, save_path
