import os
import numpy as np
import re

from motion_residual import compute_motion_residual
from mr_lbp import extract_mr_lbp_features, save_mr_lbp

preprocessed_dir = "feature_extraction/preprocessed"
output_dir = "mr_lbp"

os.makedirs(output_dir, exist_ok=True)

# Cache existing outputs ONCE
existing_outputs = set(os.listdir(output_dir))

for idx, file in enumerate(sorted(os.listdir(preprocessed_dir)), start=1):
    if not file.endswith(".npy"):
        continue

    raw_name = os.path.splitext(file)[0]

    # Hardened filename sanitizer
    video_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', raw_name).strip()

    expected_file = f"{video_name}_MRLBP.npy"

    if expected_file in existing_outputs:
        print(f"[SKIP] {video_name} already processed")
        continue

    frames = np.load(os.path.join(preprocessed_dir, file))

    mr_frames = compute_motion_residual(frames)
    feature_vector = extract_mr_lbp_features(mr_frames)

    save_path = save_mr_lbp(video_name, feature_vector)

    # 🔑 update cache after saving
    existing_outputs.add(os.path.basename(save_path))

    print(f"[{idx:03d}] {video_name}")
    print(f"      Feature shape : {feature_vector.shape}")
    print(f"      Saved at      : {save_path}\n")
