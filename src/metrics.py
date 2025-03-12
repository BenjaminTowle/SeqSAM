import os
import json
import numpy as np
import pickle
from medpy.metric import jc, dc
from PIL import Image

from medpy.metric import jc, dc
from typing import Optional, Iterable

def calc_dsc(image_0: np.array, image_1):
    if np.sum(image_0) == 0 and np.sum(image_1) == 0:
        return np.nan
    else:
        return dc(image_1, image_0)

def compute_dice(predictions: np.array, labels: np.array):
    results = []
    for pred, label in zip(predictions, labels):
        pred = np.round(np.mean((pred > 0.0).astype(np.float32), axis=0)).astype(np.int64)
        label = label.astype(np.int64)
        #results.append(
            #np.mean([np.mean([dice(target == i, pred == i) for i in range(2)]) for target in label]))
        scores = [calc_dsc(target, pred) for target in label]
        # 0.0 when undefined
        scores = [s if not np.isnan(s) else 0.0 for s in scores] 
        results.append(np.mean(scores))
        #results.append(np.mean([calc_dsc(target, pred) for target in label]))

    results = [result if not np.isnan(result) else 0.0 for result in results ]

    return results

def compute_dice_max(predictions: np.array, labels: np.array):
    results = []
    for pred, label in zip(predictions, labels):
        pred = pred > 0.0
        dice_scores = np.zeros((len(pred), len(label)))
        for i, p in enumerate(pred):
            for j, l in enumerate(label):
                dice_scores[i, j] = dice(p, l)
        max_dice_scores = np.max(dice_scores, axis=1)
        dmax = np.mean(max_dice_scores)
        results.append(dmax)

    return results

def compute_dice_nod(predictions: np.array, labels: np.array):
    results = []
    for pred, label in zip(predictions, labels):
        pred = np.round(np.mean((pred > 0.0).astype(np.float32), axis=0)).astype(np.int64)
        label = label.astype(np.int64)

        if np.sum(label) == 0:
            continue

        results.append(np.mean([calc_dsc(target, pred) for target in label if np.sum(target) > 0]))

    return results

def compute_ged(predictions: np.array, labels: np.array):
    _geds = []
    diversities = []
    for pred, label in zip(predictions, labels):
        pred = (pred > 0.0).astype(np.int64)
        _ged, diversity = generalised_energy_distance(pred, label, 1, range(1, 2))
        _geds.append(_ged)
        diversities.append(diversity)

    return _geds, diversities

def compute_blanks(predictions: np.array, labels: np.array):
    pred_blanks = []
    label_blanks = []
    for pred, label in zip(predictions, labels):
        for p, l in zip(pred, label):
            pred_blanks.append(float((np.sum(p) == 0)))
            label_blanks.append(float((np.sum(l) == 0)))
    
    return pred_blanks, label_blanks

def iou(A: np.array, B: np.array) -> float:
    intersection = np.logical_and(A, B)
    union = np.logical_or(A, B)
    if np.sum(union) == 0:
        return 0.0
    return np.sum(intersection) / np.sum(union)


def dice(A: np.array, B: np.array) -> float:
    intersection = np.logical_and(A, B)
    return 2 * np.sum(intersection) / (np.sum(A) + np.sum(B))


def dist_fct(m1: np.array, m2: np.array, nlabels: int, label_range: Iterable):
    per_label_iou = []
    for lbl in label_range:

        # assert not lbl == 0  # tmp check
        m1_bin = (m1 == lbl) * 1
        m2_bin = (m2 == lbl) * 1

        if np.sum(m1_bin) == 0 and np.sum(m2_bin) == 0:
            per_label_iou.append(1)
        elif np.sum(m1_bin) > 0 and np.sum(m2_bin) == 0 or np.sum(m1_bin) == 0 and np.sum(m2_bin) > 0:
            per_label_iou.append(0)
        else:
            per_label_iou.append(jc(m1_bin, m2_bin))

    return 1 - (sum(per_label_iou) / nlabels)


def generalised_energy_distance(sample_arr: np.array, gt_arr: np.array, nlabels: int, label_range: Iterable):
    """
    :param sample_arr: expected shape N x X x Y 
    :param gt_arr: M x X x Y 
    """
    N = sample_arr.shape[0]
    M = gt_arr.shape[0]

    d_sy = []
    d_ss = []
    d_yy = []

    for i in range(N):
        for j in range(M):
            d_sy.append(dist_fct(sample_arr[i, ...], gt_arr[j, ...], nlabels, label_range))
        for j in range(N):
            d_ss.append(dist_fct(sample_arr[i, ...], sample_arr[j, ...], nlabels, label_range))

    for i in range(M):
        for j in range(M):
            d_yy.append(dist_fct(gt_arr[i, ...], gt_arr[j, ...], nlabels, label_range))
    diversity = (1. / N ** 2) * sum(d_ss)
    return (2. / (N * M)) * sum(d_sy) - diversity - (1. / M ** 2) * sum(d_yy), diversity


def collective_insight(predictions, labels):
    cis = []
    for pred, label in zip(predictions, labels):
        pred = pred > 0.0
        
        # Combined sensitivity
        pred_union = np.sum(pred, axis=0) > 0
        label_union = np.sum(label, axis=0) > 0
        tp = np.sum(np.logical_and(pred_union, label_union))
        fn = np.sum(np.logical_and(label_union, np.logical_not(pred_union)))
        sc = tp / (tp + fn)

        # Maximum dice matching
        dice_scores = np.zeros((len(pred), len(label)))
        for i, p in enumerate(pred):
            for j, l in enumerate(label):
                dice_scores[i, j] = dice(p, l)
        max_dice_scores = np.max(dice_scores, axis=1)
        dmax = np.mean(max_dice_scores)

        # Diversity agreement
        dice_scores = []
        for i, p in enumerate(pred):
            for j, l in enumerate(pred):
                if i == j:
                    continue
                dice_scores.append(1 - dice(p, l))
        # The original paper calls this variance, but it is actually the diversity
        var_pred_min = min(dice_scores)
        var_pred_max = max(dice_scores)

        dice_scores = []
        for i, p in enumerate(label):
            for j, l in enumerate(label):
                if i == j:
                    continue
                dice_scores.append(1 - dice(p, l))
        # The original paper calls this variance, but it is actually the diversity
        if dice_scores == []:
            dice_scores = [0.0]
        var_label_min = min(dice_scores)
        var_label_max = max(dice_scores)

        var_min = abs(var_pred_min - var_label_min)
        var_max = abs(var_pred_max - var_label_max)

        da = 1 - ((var_min + var_max) / 2)

        ci = (3 * sc * dmax * da) / (sc + dmax + da)
        cis.append(ci)

    return cis

def compute_metrics(eval_pred, write_path: str = None, dataset = None): 
    predictions = eval_pred.predictions[1]
    try:
        labels = eval_pred.label_ids.astype(bool)
    except:
        labels = eval_pred.label_ids[0].astype(bool)
    non_empty_predictions = []
    non_empty_labels = []
    for label in labels:
        non_empty_label = [lab for lab in label if np.sum(lab) > 0]
        non_empty_labels.append(np.array(non_empty_label))
    #non_empty_labels = np.array(non_empty_labels)

    predictions = predictions.squeeze(1)  # Binary classification, so remove class dim
    ged, diversity = compute_ged(predictions, labels)
    _, label_diversity = compute_ged(labels, labels)
    pred_blanks, label_blanks = compute_blanks(predictions, labels)

    results = {
        "dice": compute_dice(predictions, labels),
        "dice_max": compute_dice_max(predictions, labels),
        "dice_nod": compute_dice_nod(predictions, labels),
        "ged": ged,
        "sample_diversity": diversity,
        "label_diversity": label_diversity,
        "pred_blanks": pred_blanks,
        "label_blanks": label_blanks
    }

    if write_path is not None:
        with open(write_path, "w") as f:
            json.dump(results, f)

    mean_results = {f"{name}_mean": np.mean(result) for name, result in results.items()}
    std_results = {f"{name}_std": np.std(result) for name, result in results.items()}
    results = {**mean_results, **std_results}

    return results
