"""
Stage 1 — Balanced + Leakage-Free + RBF SVM + Clean Output
"""

import numpy as np
import os
import random
import time
from datetime import datetime
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
import joblib


# ------------------------------------------------------------
# SELECT BALANCED VIDEOS
# ------------------------------------------------------------
def select_balanced_videos(auth_dir, tamp_dir, n_samples=133):

    auth_files = list(Path(auth_dir).glob("*.npy"))
    tamp_files = list(Path(tamp_dir).glob("*.npy"))

    random.seed(42)

    selected_auth = random.sample(auth_files, n_samples)
    selected_tamp = random.sample(tamp_files, n_samples)

    return selected_auth, selected_tamp


# ------------------------------------------------------------
# LOAD VIDEO FEATURES
# ------------------------------------------------------------
def load_video_features(file_list, label):

    X, y, video_ids = [], [], []

    for file in file_list:
        data = np.load(file)
        video_name = file.stem

        X.append(data)
        y.append(np.full(len(data), label))
        video_ids.extend([video_name] * len(data))

    return np.vstack(X), np.hstack(y), np.array(video_ids)


# ------------------------------------------------------------
# VIDEO-LEVEL MAJORITY VOTING
# ------------------------------------------------------------
def video_level_predictions(y_true, y_pred, video_ids):

    video_votes = defaultdict(list)
    video_labels = {}

    for yt, yp, vid in zip(y_true, y_pred, video_ids):
        video_votes[vid].append(yp)
        video_labels[vid] = yt

    final_true = []
    final_pred = []

    for vid in video_votes:
        majority = int(np.mean(video_votes[vid]) >= 0.5)
        final_true.append(video_labels[vid])
        final_pred.append(majority)

    return np.array(final_true), np.array(final_pred)


# ------------------------------------------------------------
# TRAINING PIPELINE
# ------------------------------------------------------------
def train_svm(auth_dir, tamp_dir, model_path):

    print("=" * 70)
    print("STAGE 1 — ADVANCED SVM TRAINING")
    print("=" * 70)
    print("Start Time:", datetime.now(), "\n")

    # 1️⃣ Balanced selection
    auth_videos, tamp_videos = select_balanced_videos(auth_dir, tamp_dir, 133)

    # 2️⃣ Video-level split
    auth_train, auth_test = train_test_split(auth_videos, test_size=0.2, random_state=42)
    tamp_train, tamp_test = train_test_split(tamp_videos, test_size=0.2, random_state=42)

    # 3️⃣ Load training
    X_auth_train, y_auth_train, vid_auth_train = load_video_features(auth_train, 0)
    X_tamp_train, y_tamp_train, vid_tamp_train = load_video_features(tamp_train, 1)

    X_train = np.vstack([X_auth_train, X_tamp_train])
    y_train = np.hstack([y_auth_train, y_tamp_train])

    # 4️⃣ Load testing
    X_auth_test, y_auth_test, vid_auth_test = load_video_features(auth_test, 0)
    X_tamp_test, y_tamp_test, vid_tamp_test = load_video_features(tamp_test, 1)

    X_test = np.vstack([X_auth_test, X_tamp_test])
    y_test = np.hstack([y_auth_test, y_tamp_test])
    vid_test = np.concatenate([vid_auth_test, vid_tamp_test])

    # 5️⃣ Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/svm_stage1_scaler.pkl")

    # 6️⃣ RBF SVM GridSearch
    param_grid = {
        'C': [1, 10],
        'gamma': ['scale', 0.01]
    }

    grid = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced'),
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )

    start = time.time()
    grid.fit(X_train, y_train)
    end = time.time()

    print("\nGridSearch completed in {:.2f} minutes".format((end - start)/60))
    print("Best parameters:", grid.best_params_, "\n")

    svm = grid.best_estimator_

    # ================= FRAME-LEVEL =================
    y_pred_frame = svm.predict(X_test)

    print("FRAME-LEVEL RESULTS")
    print("-" * 70)
    print(f"Accuracy : {accuracy_score(y_test, y_pred_frame):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_frame):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred_frame):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred_frame):.4f}")

    # ================= VIDEO-LEVEL =================
    y_true_video, y_pred_video = video_level_predictions(
        y_test, y_pred_frame, vid_test
    )

    print("\nVIDEO-LEVEL RESULTS")
    print("-" * 70)
    print(f"Accuracy : {accuracy_score(y_true_video, y_pred_video):.4f}")
    print(f"Precision: {precision_score(y_true_video, y_pred_video):.4f}")
    print(f"Recall   : {recall_score(y_true_video, y_pred_video):.4f}")
    print(f"F1-score : {f1_score(y_true_video, y_pred_video):.4f}")

    joblib.dump(svm, model_path)
    print("\nModel saved to:", model_path)
    print("End Time:", datetime.now())


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    print("Script started\n")

    AUTHENTIC_DIR = "mr_lbp/authentic"
    TAMPERED_DIR = "mr_lbp/tampered"
    MODEL_PATH = "models/svm_stage1_model.pkl"

    train_svm(AUTHENTIC_DIR, TAMPERED_DIR, MODEL_PATH)
