import typing as tp
from collections import defaultdict

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import PathConfig
from scripts.load_labels import load_ground_truth_labels, load_predicted_labels, load_combined_predicted_labels, \
    ACTIVE_LABELS, NEGLIGIBLE_LABELS, PASSIVE_LABELS


_T_LABEL = tp.Tuple[str, float, float]
TOLERANCE = 5
MINIMUM_LABELS = 3


def get_intersections(boundary1: tp.Tuple[float, float], labels: tp.List[_T_LABEL], tolerance: int = 0
                      ) -> tp.List[_T_LABEL]:
    intersections = []
    for label, *boundary2 in labels:
        if boundary1[1] + tolerance > boundary2[0] and boundary1[0] - tolerance < boundary2[1]:
            intersections.append((label, *boundary2))
    return intersections


def create_single_confusion_matrix(predictions: tp.List[_T_LABEL], ground_truth: tp.List[_T_LABEL], min_num: int = 1):
    predictions_copy = predictions.copy()
    confusion_matrix = np.zeros((2, 2))
    for label, *boundary in ground_truth:
        assert label in ACTIVE_LABELS + PASSIVE_LABELS + NEGLIGIBLE_LABELS, \
            f"Label: {label} is not supported, add it to above constraints"
        intersections = get_intersections(boundary, predictions_copy, tolerance=TOLERANCE)
        if label not in NEGLIGIBLE_LABELS:
            is_prediction = 0 if len(intersections) >= min_num else 1
            is_actual = 0 if label in ACTIVE_LABELS else 1
            confusion_matrix[is_prediction, is_actual] += 1
        [predictions_copy.remove(elem) for elem in intersections]

    for detector, *boundary in predictions_copy.copy():
        detections = get_intersections(boundary, predictions_copy, tolerance=0)
        if len(detections) >= min_num:
            confusion_matrix[0, 1] += 1
            [predictions_copy.remove(elem) for elem in detections]
    return confusion_matrix


def create_confusion_matrices_by_detector(predictions_by_file_id: tp.Dict, ground_truth_by_file_id: tp.Dict):
    confusion_matrix_by_detector = defaultdict(lambda: np.zeros((2, 2)))
    for file_id, predictions in tqdm(list(predictions_by_file_id.items()), desc="Calculating confusion matrix"):
        predictions_by_detector = defaultdict(list)
        [predictions_by_detector[prediction[0]].append(prediction) for prediction in predictions]
        for detector, detector_prediction in predictions_by_detector.items():
            confusion_matrix = create_single_confusion_matrix(detector_prediction, ground_truth_by_file_id[file_id])
            confusion_matrix_by_detector[detector] += confusion_matrix
    return dict(confusion_matrix_by_detector)


def create_combined_detector_matrix(predictions_by_file_id: tp.Dict, ground_truth_by_file_id: tp.Dict, min_num: int):
    combined_matrix = np.zeros((2, 2))
    for file_id, pred in tqdm(list(predictions_by_file_id.items()), desc="Calculating combined confusion matrix"):
        confusion_matrix = create_single_confusion_matrix(pred, ground_truth_by_file_id[file_id], min_num=min_num)
        combined_matrix += confusion_matrix
    return {f"Combined M={min_num}": combined_matrix}


def plot_confusion_matrices(confusion_matrix_by_detector: tp.Dict, model_label: str):
    for detector, confusion_matrix in confusion_matrix_by_detector.items():
        sn.heatmap(confusion_matrix, cmap="Blues", annot=True, annot_kws={"size": 12}, fmt='g')
        plt.title(f"Detector: '{detector.replace('_', ' ')}', T={TOLERANCE} Model: {model_label.upper()}")
        plt.xlabel("Actual value")
        plt.ylabel("Predicted value")
        plt.xticks([0.5, 1.5], ["True", "False"])
        plt.yticks([0.5, 1.5], ["True", "False"])
        PathConfig.mkdir(PathConfig.CM_PATH)
        plt.savefig(f"{PathConfig.CM_PATH}/CM_{detector}_T{TOLERANCE}_{model_label}.png")
        plt.show()


def calculate_metrics(confusion_matrix_by_detector: tp.Dict, model_label: str):
    calculated_metrics = [
        {
            "name": detector,
            "confusion_matrix": cm,
            "accuracy": round((cm[0, 0] + cm[1, 1]) / cm.sum(), 2),
            "recall": round(cm[0, 0] / cm[:, 0].sum(), 2),
            "precision": round(cm[0, 0] / cm[0, :].sum(), 2),
        } for detector, cm in confusion_matrix_by_detector.items()
    ]

    print(f"Tolerance = {TOLERANCE}s | Model: {model_label.upper()}")
    for metrics_dict in calculated_metrics:
        print("- " * 30)
        print(f"Metrics for {metrics_dict['name']} detector:")
        print(f"Confusion matrix: \n{metrics_dict['confusion_matrix']}")
        print(f"Accuracy: {metrics_dict['accuracy']}")
        print(f"Recall: {metrics_dict['recall']}")
        print(f"Precision: {metrics_dict['precision']}")
        print()
    return calculated_metrics


def print_best_metrics(calculated_metrics: tp.List[tp.Dict]):
    best_accuracy = max(calculated_metrics, key=lambda x: x["accuracy"])
    print(f"Best accuracy detector: {best_accuracy['name']} accuracy: {best_accuracy['accuracy']}")

    best_recall = max(calculated_metrics, key=lambda x: x["recall"])
    print(f"Best recall detector: {best_recall['name']} recall: {best_recall['recall']}")

    best_precision = max(calculated_metrics, key=lambda x: x["precision"])
    print(f"Best precision detector: {best_precision['name']} precision: {best_precision['precision']}")


def calculate_detectors_correlation(predictions_by_file_id: tp.Dict, precision: int = 1):
    labels = list(np.unique([prediction[0] for predictions in predictions_by_file_id.values() for prediction in predictions]))

    corr_matrix = None
    for file_id, predictions in tqdm(list(predictions_by_file_id.items()), desc="Calculating detectors correlation"):
        vector_length = np.ceil(np.array(predictions)[:, -2:].astype(float)).max().astype(int) if predictions else 1
        time_vector = np.arange(0, vector_length - 1, 10**(-precision))
        df = pd.DataFrame(np.zeros((len(time_vector), len(labels))), columns=labels)
        for detector, t0, t1 in predictions:
            df[detector][round(t0 * 10**precision): round(t1 * 10**precision)] = 1
        df_corr = df.corr().fillna(1)
        corr_matrix = df_corr if corr_matrix is None else corr_matrix + df_corr
    return corr_matrix / len(predictions_by_file_id)


def plot_correlation_matrix(correlation_matrix: pd.DataFrame, model_label: str):
    y_ticks = [elem.replace('_', '\n') for elem in correlation_matrix.index]
    x_ticks = [elem.replace('_', '\n') for elem in correlation_matrix.columns]
    sn.set(font_scale=0.8)
    sn.heatmap(correlation_matrix, cmap="Blues", annot=True, yticklabels=y_ticks, xticklabels=x_ticks)
    plt.xticks(rotation=0)
    plt.title(f"Correlation matrix of application detectors using {model_label.upper()} model", fontdict={"size": 14})
    plt.tight_layout()
    PathConfig.mkdir(PathConfig.CM_PATH)
    plt.savefig(f"{PathConfig.CM_PATH}/Correlation_matrix_{model_label}.png")
    plt.show()


if __name__ == "__main__":
    GROUND_TRUTH = load_ground_truth_labels()

    # PREDICTED = load_combined_predicted_labels()
    # for n in range(1, 30):
    #     COMBINED_MATRIX = create_combined_detector_matrix(PREDICTED, GROUND_TRUTH, n)
    #     plot_confusion_matrices(COMBINED_MATRIX, "Combined")
    #     calculate_metrics(COMBINED_MATRIX, "Combined")
    # CORRELATION = calculate_detectors_correlation(PREDICTED)
    # plot_correlation_matrix(CORRELATION, "Combined")

    for MODEL_LABEL in ["cnn", "rnn"]:
        try:
            PREDICTED = load_predicted_labels(MODEL_LABEL)
        except FileNotFoundError:
            continue

        # Confusion matrices by detectors
        CONFUSION_MATRICES = create_confusion_matrices_by_detector(PREDICTED, GROUND_TRUTH)
        plot_confusion_matrices(CONFUSION_MATRICES, MODEL_LABEL)
        calculate_metrics(CONFUSION_MATRICES, MODEL_LABEL)

        # Combined confusion matrix from all detectors
        METRICS_LIST = []
        for n in range(1, 6):
            COMBINED_MATRIX = create_combined_detector_matrix(PREDICTED, GROUND_TRUTH, n)
            plot_confusion_matrices(COMBINED_MATRIX, MODEL_LABEL)
            METRIC = calculate_metrics(COMBINED_MATRIX, MODEL_LABEL)
            METRICS_LIST.extend(METRIC)
        print_best_metrics(METRICS_LIST)

        # Correlation matrix between detectors
        CORRELATION = calculate_detectors_correlation(PREDICTED)
        plot_correlation_matrix(CORRELATION, MODEL_LABEL)
