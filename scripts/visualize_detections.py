import typing as tp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

from config import PathConfig
from scripts.load_labels import load_info, load_ground_truth_labels, load_predicted_labels, load_valence_arousal, \
    load_combined_predicted_labels, ACTIVE_LABELS


def plot_detections(ground_truth_by_file_id: tp.Dict, info: tp.Dict,
                    predictions_by_file_id: tp.Dict, valence: tp.Dict, arousal: tp.Dict, model_label: str):
    PathConfig.mkdir(PathConfig.PLOTS_PATH)

    for file_id, predictions in tqdm(list(predictions_by_file_id.items()), desc="Plotting..."):
        labels = list(np.unique([prediction[0] for prediction in predictions]))
        labels.sort(key=lambda x: (x[-9] if x.endswith("naive") else x[-3], x[0]), reverse=True)
        fig, (a0, a1) = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]})
        fig.set_size_inches(28.5, 10.5)
        plt.style.use("seaborn-whitegrid")

        video_duration = info[file_id][0]

        # A0
        # Plot label names
        for i, label in enumerate(labels):
            a0.text(video_duration / 20, i + 0.5, label, horizontalalignment='center', verticalalignment='center')

        # Plot model predictions boxes
        for label, start_time, end_time in predictions:
            x1 = [start_time, end_time]
            label_index = labels.index(label)
            y1 = [label_index + 0.05] * 2
            y2 = [label_index - 0.05 + 1] * 2
            color = cm.get_cmap('winter')((label_index + 1)/len(labels))
            a0.fill_between(x1, y1, y2=y2, color=color, label=label)

        # Plot grand truth columns
        bars = []
        for label, start_time, end_time in ground_truth_by_file_id[file_id]:
            if label not in ACTIVE_LABELS:
                continue
            intersections_num = sum([x0 < start_time < x1 for (x0, x1) in bars])
            x1 = [start_time, end_time]
            y1 = [0, 0]
            y2 = [len(labels) + 0.1 + (intersections_num * 0.3)] * 2
            a0.fill_between(x1, y1, y2=y2, color="red", label=label, alpha=0.1)
            a0.text(np.mean(x1), y2[0] + 0.1, label, horizontalalignment='center', verticalalignment='center')
            bars.append((x1[0] - 20, x1[1] + 20))

        a0.set_xlim(left=0, right=video_duration)
        a0.set_xticks(np.arange(0, video_duration, 10))
        a0.set_xlabel("Seconds")

        # A1
        # Plot valence and arousal
        a1.plot(valence[file_id], label="Valence")
        a1.plot(arousal[file_id], label="Arousal")

        a1.set_xlim(left=-100, right=len(arousal[file_id]))
        a1.set_xlabel("Frames")
        fig.suptitle(f"Predictions for {file_id} using {model_label.upper()} model")
        plt.legend()
        plt.savefig(f"{PathConfig.PLOTS_PATH}/{file_id}_{model_label}.png")
        plt.show()


if __name__ == "__main__":
    GROUND_TRUTH = load_ground_truth_labels()
    DF_INFO = load_info()

    COMBINED_PREDICTIONS = load_combined_predicted_labels()
    VALENCE, AROUSAL = load_valence_arousal("rnn")
    plot_detections(GROUND_TRUTH, DF_INFO, COMBINED_PREDICTIONS, VALENCE, AROUSAL, "Combined")

    for MODEL_LABEL in ["cnn", "rnn", "cnn_naive", "rnn_naive"]:
        PREDICTED = load_predicted_labels(MODEL_LABEL)
        VALENCE, AROUSAL = load_valence_arousal(MODEL_LABEL.split("_")[0])
        plot_detections(GROUND_TRUTH, DF_INFO, PREDICTED, VALENCE, AROUSAL, MODEL_LABEL)
