import os
from preprocess import preprocess_video

DATASETS = {
    "ORIGINAL": r"H:\project_data\original",
    "FRAME DUPLICATION": r"H:\project_data\tampered_fd",
    "FRAME INSERTION": r"H:\project_data\tampered_fi"
}

for dataset_name, dataset_path in DATASETS.items():
    print(f"{dataset_name} Dataset – Preprocessing")
    print("-" * 90)

    for idx, video_name in enumerate(os.listdir(dataset_path), start=1):

        if not video_name.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

        video_path = os.path.join(dataset_path, video_name)

        try:
            frames, total_frames, fps, duration, save_path = preprocess_video(video_path)

            print(
                f"[{idx:02d}] {video_name}\n"
                f"     Frames   : {total_frames}\n"
                f"     FPS      : {fps:.2f}\n"
                f"     Duration : {duration:.2f} sec\n"
                f"     Saved at : {save_path}\n"
                f"     Status   : SUCCESS\n"
            )

        except Exception as e:
            print(
                f"[{idx:02d}] {video_name}\n"
                f"     Status   : FAILED\n"
                f"     Error    : {str(e)}\n"
            )

print("\n" + "-" * 90)
print("Completed")
