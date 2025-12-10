# code/evaluation.py

"""
Evaluation utilities for Dynamic Earth Net experiments.
Includes functions for calculating accuracy metrics, confusion matrices, and saving results.
"""

from typing import Optional
import numpy as np
import pandas as pd

def calc_acc_metrices(cm: np.array, path: Optional[str] = None, excel: bool = True, print_cons: bool = False):
    """
    Calculate accuracy metrics (F1, precision, recall, IoU, OA, etc.) from a confusion matrix.
    Args:
        cm: Confusion matrix (2D numpy array).
        path: Optional path to save results.
        excel: If True, save results as Excel file.
        print_cons: If True, print confusion matrix and metrics.
    Returns:
        Tuple of (f1_scores, precision, recall, iou, oa, support).
    """

    # True Positives, False Positives, False Negatives, Support
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    support = np.sum(cm, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0)
        f1_scores = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0)
    oa = np.sum(tp) / np.sum(cm) if np.sum(cm) > 0 else 0

    if print_cons:
        print("Confusion Matrix:\n", cm)
        print("F1 scores:", f1_scores)
        #print("Precision:", precision)
        #print("Recall:", recall)
        #print("IoU:", iou)
        print("Overall Accuracy (OA):", oa)

    if path:
        results = {
            "F1": f1_scores,
            "Precision": precision,
            "Recall": recall,
            "IoU": iou,
            "Support": support
        }
        df = pd.DataFrame(results)
        #if excel:
        #    df.to_excel(path + ".xlsx")
        #else:
        df.to_csv(path + ".csv")
        np.save(path + "_cm.npy", cm)


    return f1_scores, iou, precision, recall, oa, support

def save_val_score(path: str, epoch: int, oa: np.ndarray, prec: np.ndarray, rec: np.ndarray):
    """
        Save validation metrics to a text file.
        Args:
            path: Path to the output text file.
            epoch: Current epoch number.
            oa: Overall accuracy array.
            prec: Precision array.
            rec: Recall array.
        """
    with open(path, "a") as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"OA: {oa}\n")
        f.write(f"Precision: {prec}\n")
        f.write(f"Recall: {rec}\n")
        f.write("\n")

def calc_OA(cm: np.ndarray):
    """
    Calculate overall accuracy (OA) from a confusion matrix.
    Args:
        cm: Confusion matrix (2D numpy array).
    Returns:
        Overall accuracy as a float.
    """
    tp = np.diag(cm)
    total = np.sum(cm)
    return float(np.sum(tp)) / total if total > 0 else 0.0

def calc_mean_F1(cm: np.ndarray):
    """
        Calculate mean F1 score from a confusion matrix.
        Args:
            cm: Confusion matrix (2D numpy array).
        Returns:
            Mean F1 score as a float.
    """
    if np.sum(cm) == 0: return 0
    fscores = []
    for i in range(len(cm)):
        support_current = np.sum(cm[i, :])
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = support_current - TP
        prec = TP / float(TP + FP)
        recall = TP / float(TP + FN)
        f_score = 2 * prec * recall / (prec + recall)
        fscores.append(f_score)
    return float(np.mean(fscores))

def calculate_conf_matrix(preds: np.ndarray, labels: np.ndarray, classes: np.ndarray, ignore_class: int = -1):
    """
    Calculate confusion matrix for predictions and labels.
    Args:
        preds: Predicted labels (flattened array).
        labels: Ground truth labels (flattened array).
        classes: Array of class indices.
    Returns:
        Confusion matrix as a 2D numpy array.
    """
    # test of labels are integers, and if not convert them
    if not np.issubdtype(labels.dtype, np.integer):
        labels = labels.astype(np.int64)
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(labels.flatten(), preds.flatten()):
        if t < n_classes and p < n_classes:
            try:
                cm[t, p] += 1
            except IndexError:
                print("IndexError in confusion matrix calculation")
                print('t: ', t, 'p: ', p, 'n_classes: ', n_classes)
    #if ignore_class != -1:
    #    # remove line and column of ignored class:
    #    cm = np.delete(cm, ignore_class, axis=0)
    #    cm = np.delete(cm, ignore_class, axis=1)
        #cm = cm[1:,1:]
    return cm

def comp_confmat(actual, predicted, classes): # todo: delete if calcualte_conf_amtr is working
    confmat = np.zeros((len(classes), len(classes)))
    for i in range(len(classes)):
        for j in range(len(classes)):
           confmat[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))
    return confmat

def calc_IoU(CM: np.ndarray):
    """
    Calculate the Intersection over Union (IoU) for each class from a confusion matrix.
    Args:
        CM: Confusion matrix as a 2D numpy array (shape: [num_classes, num_classes]).
    Returns:
        List of IoU values for each class (float).
    """
    if CM.size == 0 or len(CM.shape) != 2 or CM.shape[0] != CM.shape[1]:
        raise ValueError("Input confusion matrix must be a non-empty square 2D array.")

    IoUs = []
    num_classes = CM.shape[0]
    for i in range(num_classes):
        support_current = np.sum(CM[i, :])
        TP = CM[i, i]
        FP = np.sum(CM[:, i]) - TP
        FN = support_current - TP
        denominator = TP + FP + FN
        IoU = TP / float(denominator) if denominator > 0 else 0.0
        IoUs.append(IoU)
    return IoUs