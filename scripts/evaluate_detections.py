import typing as tp
from collections import defaultdict

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import PathConfig
from scripts.load_labels import load_ground_truth_labels, load_predicted_labels, LABELS


_T_LABEL = tp.Tuple[str, float, float]
TOLERANCE = 5


def get_intersections(boundary1: tp.Tuple[float, float], labels: tp.List[_T_LABEL], tolerance: int = 0):
    intersections = []
    for label, *boundary2 in labels:
        if boundary1[1] + tolerance > boundary2[0] and boundary1[0] - tolerance < boundary2[1]:
            intersections.append((label, *boundary2))
    return intersections


def create_single_confusion_matrix(predictions: tp.List[_T_LABEL], ground_truth: tp.List[_T_LABEL]):
    confusion_matrix = np.zeros((2, 2))
    for detector, *boundary in predictions:
        intersections = get_intersections(boundary, ground_truth, tolerance=TOLERANCE)
        if not intersections:
            confusion_matrix[0, 1] += 1
        else:
            for label, *_ in intersections:
                confusion_matrix[0, 1 if label == "neutral" else 0] += 1

    for label, *boundary in ground_truth:
        intersections = get_intersections(boundary, predictions)
        if not intersections:
            confusion_matrix[1, 1 if label == "neutral" else 0] += 1
    return confusion_matrix


def create_cumulated_confusion_matrices(predictions_by_file_id: tp.Dict, ground_truth_by_file_id: tp.Dict):
    confusion_matrix_by_detector = defaultdict(lambda: np.zeros((2, 2)))
    for file_id, predictions in tqdm(list(predictions_by_file_id.items()), desc="Calculating confusion matrix"):
        predictions_by_detector = defaultdict(list)
        [predictions_by_detector[prediction[0]].append(prediction) for prediction in predictions]
        for detector, detector_prediction in predictions_by_detector.items():
            confusion_matrix = create_single_confusion_matrix(detector_prediction, ground_truth_by_file_id[file_id])
            confusion_matrix_by_detector[detector] += confusion_matrix
    return dict(confusion_matrix_by_detector)


def plot_confusion_matrices(confusion_matrix_by_detector: tp.Dict):
    for detector, confusion_matrix in confusion_matrix_by_detector.items():
        sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 12}, fmt='g')
        plt.title(f"Detector: '{detector.replace('_', ' ')}', T={TOLERANCE}")
        plt.xlabel("Actual value")
        plt.ylabel("Predicted value")
        plt.xticks([0.5, 1.5], ["True", "False"])
        plt.yticks([0.5, 1.5], ["True", "False"])
        plt.savefig(f"{PathConfig.PLOTS_PATH}/CM_{detector}_T{TOLERANCE}.png")
        plt.show()


def calculate_metrics(confusion_matrix_by_detector: tp.Dict):
    print(f"Tolerance = {TOLERANCE}s")
    for detector, cm in confusion_matrix_by_detector.items():
        print("- " * 30)
        print(f"Metrics for {detector} detector:")
        print(f"Confusion matrix: \n{cm}")
        _tp = cm[0, 0]
        print(f"Accuracy: {round((_tp + cm[1, 1]) / cm.sum(), 2)}")
        print(f"Recall: {round(_tp / cm[:, 0].sum(), 2)}")
        print(f"Precision: {round(_tp / cm[0, :].sum(), 2)}")
        print()


def calculate_detectors_correlation(predictions_by_file_id: tp.Dict, precision: int = 1):
    corr_matrix = None
    for file_id, predictions in tqdm(list(predictions_by_file_id.items()), desc="Calculating detectors correlation"):
        vector_length = np.ceil(np.array(predictions)[:, -2:].astype(float)).max().astype(int) if predictions else 1
        time_vector = np.arange(0, vector_length - 1, 10**(-precision))
        df = pd.DataFrame(np.zeros((len(time_vector), len(LABELS))), columns=LABELS)
        for detector, t0, t1 in predictions:
            df[detector][round(t0 * 10**precision) : round(t1 * 10**precision)] = 1
        df_corr = df.corr().fillna(1)
        corr_matrix = df_corr if corr_matrix is None else corr_matrix + df_corr
    return corr_matrix / len(predictions_by_file_id)


def plot_correlation_matrix(correlation_matrix: pd.DataFrame):
    y_ticks = [elem.replace('_', '\n') for elem in correlation_matrix.index]
    x_ticks = [elem.replace('_', '\n') for elem in correlation_matrix.columns]
    sn.set(font_scale=0.8)
    sn.heatmap(correlation_matrix, annot=True, yticklabels=y_ticks, xticklabels=x_ticks)
    plt.xticks(rotation=0)
    plt.title("Correlation matrix of application detectors", fontdict={"size": 14})
    plt.tight_layout()
    plt.savefig(f"{PathConfig.PLOTS_PATH}/Correlation_matrix.png")
    plt.show()


if __name__ == "__main__":
    PREDICTED = load_predicted_labels()
    GROUND_TRUTH = load_ground_truth_labels()
    CONFUSION_MATRICES = create_cumulated_confusion_matrices(PREDICTED, GROUND_TRUTH)
    CORRELATION = calculate_detectors_correlation(PREDICTED)
    plot_confusion_matrices(CONFUSION_MATRICES)
    plot_correlation_matrix(CORRELATION)
    calculate_metrics(CONFUSION_MATRICES)

