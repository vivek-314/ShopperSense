"""Pure evaluation and validation-only selection functions."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from .config import MODEL_ORDER


def metrics(y, scores, threshold=.5):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError('Labels must be a nonempty binary array')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be finite and within [0,1]')
    if len(y) != len(scores) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Scores must be finite, in [0,1], and aligned to labels')
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        'threshold': float(threshold), 'support': len(y), 'purchases': int(y.sum()),
        'predicted_purchases': int(predicted.sum()),
        'precision': float(precision_score(y, predicted, zero_division=0)),
        'recall': float(recall_score(y, predicted, zero_division=0)),
        'f1': float(f1_score(y, predicted, zero_division=0)),
        'ap': float(average_precision_score(y, scores)) if y.sum() else None,
        'roc_auc': float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        'accuracy': float(accuracy_score(y, predicted)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'precision_defined': bool(predicted.sum()), 'recall_defined': bool(y.sum()),
    }


def choose_model(validation_records):
    eligible = [r for r in validation_records if r['feature_set'] == 'full' and r['model'] in MODEL_ORDER]
    top = max(r['ap'] for r in eligible)
    return min((r for r in eligible if top-r['ap'] <= .005), key=lambda r: MODEL_ORDER.index(r['model']))['model']


def choose_threshold(validation_y, validation_scores):
    grid = [metrics(validation_y, validation_scores, step/100) for step in range(5, 96)]
    best = max(grid, key=lambda r: (r['f1'], r['threshold']))
    return best['threshold'], grid
