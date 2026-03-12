import numpy as np

def compute_motion_residual(frames):
    """
    frames: (N, H, W)
    returns: (N-1, H, W) absolute motion residuals
    """
    motion_residuals = []

    for i in range(1, len(frames)):
        diff = np.abs(
            frames[i].astype(np.int16) - frames[i-1].astype(np.int16)
        )
        motion_residuals.append(diff.astype(np.uint8))

    return np.array(motion_residuals)
