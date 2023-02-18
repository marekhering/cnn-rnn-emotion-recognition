import os
import typing as tp
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

from config import PathConfig


POINT_EVENTS = ["local_rapid_deprecation", "global_rapid_deprecation"]
STATE_EVENTS = ["global_deviation", "local_deviation", "global_sigmoid_deviation", "local_sigmoid_deviation"]
LABELS = POINT_EVENTS + STATE_EVENTS
POINTS_EVENT_TIME = 1


def load_ground_truth_labels():
    labeled_emotions = defaultdict(list)
    for filename in tqdm(os.listdir(PathConfig.EVAL_VIDEOS_PATH), desc="Load ground truth labels"):
        filename_split = filename.split('_PLAYER1_')
        file_id = filename_split[0].replace('-zgoda', '')
        emotion, seconds = filename_split[1][:-4].split("_")[1:]
        start_time, end_time = [float(s) for s in seconds.split('-')]
        assert all([len(emotion), isinstance(start_time, float), isinstance(end_time, float)])
        labeled_emotions[file_id].append((emotion, start_time, end_time))
    return dict(labeled_emotions)


def load_predicted_labels():
    def convert_line(_line: str):
        _line_split = _line.split("\t\t")
        return float(_line_split[0]), _line_split[1]

    detections = {}
    for filename in tqdm(os.listdir(PathConfig.OUTPUT_VIDEOS_PATH), desc="Load predicted labels"):
        with open(f"{PathConfig.OUTPUT_VIDEOS_PATH}/{filename}") as f:
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


def plot_detections(grand_truth_by_file_id: tp.Dict, predicted: tp.Dict):
    PathConfig.mkdir(PathConfig.PLOTS_PATH)
    for file_id, predictions in tqdm(list(predicted.items()), desc="Plotting..."):
        fig = plt.gcf()
        fig.set_size_inches(28.5, 10.5)
        plt.style.use("seaborn-whitegrid")

        # Plot label names
        max_x = max([p[2] for p in predictions + grand_truth_by_file_id[file_id]])
        for i, label in enumerate(LABELS):
            plt.text(max_x / 20, i + 0.5, label, horizontalalignment='center', verticalalignment='center')

        # Plot model predictions boxes
        for label, start_time, end_time in predictions:
            x1 = [start_time, end_time]
            label_index = LABELS.index(label)
            y1 = [label_index + 0.05] * 2
            y2 = [label_index - 0.05 + 1] * 2
            color = cm.get_cmap('winter')((label_index + 1)/len(LABELS))
            plt.fill_between(x1, y1, y2=y2, color=color, label=label)

        # Plot grand truth columns
        bars = []
        for label, start_time, end_time in grand_truth_by_file_id[file_id]:
            intersections_num = sum([x0 < start_time < x1 for (x0, x1) in bars])
            x1 = [start_time, end_time]
            y1 = [0, 0]
            y2 = [len(LABELS) + 0.1 + (intersections_num * 0.3)] * 2
            plt.fill_between(x1, y1, y2=y2, color="red", label=label, alpha=0.1)
            plt.text(np.mean(x1), y2[0] + 0.1, label, horizontalalignment='center', verticalalignment='center')
            bars.append((x1[0] - 20, x1[1] + 20))

        plt.xlim(left=0)
        plt.title(f"Predictions for {file_id}")
        plt.xlabel("Seconds")
        plt.savefig(f"{PathConfig.PLOTS_PATH}/{file_id}.png")
        plt.show()


if __name__ == "__main__":
    GRAND_TRUTH = load_ground_truth_labels()
    PREDICTED = load_predicted_labels()
    plot_detections(GRAND_TRUTH, PREDICTED)
