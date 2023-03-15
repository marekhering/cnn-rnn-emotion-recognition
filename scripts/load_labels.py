import os
import typing as tp
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from config import PathConfig


POINT_EVENTS = ["global_rapid_deprecation", "local_rapid_deprecation"]
STATE_EVENTS = ["global_deviation", "global_sigmoid_deviation", "local_deviation", "local_sigmoid_deviation"]
LABELS = POINT_EVENTS + STATE_EVENTS
POINTS_EVENT_TIME = 1


def load_ground_truth_labels():
    labeled_emotions = defaultdict(list)
    for filename in tqdm(os.listdir(PathConfig.EVAL_VIDEOS_PATH), desc="Loading ground truth labels"):
        filename_split = filename.split('_PLAYER1_')
        file_id = filename_split[0].replace('-zgoda', '')
        emotion, seconds = filename_split[1][:-4].split("_")[1:]
        start_time, end_time = [float(s) for s in seconds.split('-')]
        assert all([len(emotion), isinstance(start_time, float), isinstance(end_time, float)])
        labeled_emotions[file_id].append((emotion, start_time, end_time))
    return dict(labeled_emotions)


def load_info() -> tp.Dict:
    info = {}
    for filename in tqdm(os.listdir(PathConfig.VIDEOS_PATH), desc="Getting videos info"):
        filepath = f"{PathConfig.VIDEOS_PATH}/{filename}"
        filename_split = filename.split("_")
        file_id = f"{filename_split[0]}_{filename_split[1][0]}"
        video = cv2.VideoCapture(filepath)
        fps = video.get(cv2.CAP_PROP_FPS)
        if not fps:
            continue
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        info[file_id] = (duration, frame_count, fps)
    return info


def load_predicted_labels(model_label: str = ""):
    def convert_line(_line: str):
        _line_split = _line.split("\t\t")
        return float(_line_split[0]), _line_split[1]

    detections = {}
    output_path = f"{PathConfig.OUTPUT_VIDEOS_PATH}_{model_label}" if model_label else PathConfig.OUTPUT_VIDEOS_PATH
    for filename in tqdm(os.listdir(output_path), desc="Loading predicted labels"):
        with open(f"{output_path}/{filename}") as f:
            lines = f.readlines()

        result = []
        for i, line1 in enumerate(lines):
            t0, label1 = convert_line(line1)
            if label1 in POINT_EVENTS:
                result.append((label1, t0, t0 + POINTS_EVENT_TIME))
            if label1 in STATE_EVENTS:
                for line2 in lines[i + 1:]:
                    t1, label2 = convert_line(line2)
                    if label1 == label2:
                        result.append((label1, t0, t1))
                        lines.remove(line2)
                        break

        filename_split = filename.split('_')
        file_id = f"{filename_split[0]}_{filename_split[1][0]}"
        detections[file_id] = result
    return detections


def load_valence_arousal(model_label: str = ""):
    valence, arousal = {}, {}
    output_path = f"{PathConfig.OUTPUT_VA_PATH}_{model_label}" if model_label else PathConfig.OUTPUT_VIDEOS_PATH
    for filename in os.listdir(output_path):
        data = np.loadtxt(f"{output_path}/{filename}")
        filename = filename.split(".")[0]
        filename_split = filename.split("_")
        file_id = f"{filename_split[0]}_{filename_split[1][0]}"
        if filename.endswith("arousal"):
            arousal[file_id] = data
        elif filename.endswith("valence"):
            valence[file_id] = data
    return valence, arousal

