import numpy as np
from skimage.feature import local_binary_pattern
import os

P = 8
R = 1
BINS = 256

def extract_mr_lbp_features(mr_frames):
    """
    mr_frames: (N-1, H, W)
    returns: (N-1, 256) Feature Map (Matrix)
    """
    feature_map = []

    for frame in mr_frames:
        lbp = local_binary_pattern(frame, P, R, method='default')
        
        hist, _ = np.histogram(lbp.ravel(), bins=BINS, range=(0, BINS))
        
        sum_h = np.sum(hist)
        hist_norm = hist / (sum_h if sum_h > 0 else 1)
        
        feature_map.append(hist_norm)

    return np.array(feature_map, dtype=np.float32)



def save_mr_lbp(video_name, feature_vector):
    save_dir = "mr_lbp"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{video_name}_MRLBP.npy")
    np.save(save_path, feature_vector.astype(np.float32))
    return save_path
