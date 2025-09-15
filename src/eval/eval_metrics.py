import json
import csv
from typing import Dict, List
from collections import Counter

from sklearn.metrics import precision_recall_fscore_support, classification_report


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    out = {
        'per_class': report,
        'macro_precision': float(report['macro avg']['precision']),
        'macro_recall': float(report['macro avg']['recall']),
        'macro_f1': float(report['macro avg']['f1-score']),
    }
    # LMI (toy): log-odds mutual information surrogate
    # Define LMI for binary case as: sum_c p(c) * log( (tp_c + 1)/(fp_c + 1) )
    # This is a placeholder if research requires; adapt as needed.
    counts = Counter()
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            counts['tp1'] += 1
        elif yt == 0 and yp == 1:
            counts['fp1'] += 1
        if yt == 0 and yp == 0:
            counts['tp0'] += 1
        elif yt == 1 and yp == 0:
            counts['fp0'] += 1
    import math
    lmi_bin = 0.0
    total = len(y_true) if y_true else 1
    for c in [0, 1]:
        tp = counts.get(f'tp{c}', 0)
        fp = counts.get(f'fp{c}', 0)
        pc = (tp + fp) / total if total else 0
        lmi_bin += pc * math.log((tp + 1) / (fp + 1))
    out['LMI'] = float(lmi_bin)
    return out


def save_metrics_to_csv_json(metrics: Dict, csv_path: str, json_path: str):
    # Save JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    # Save CSV: flatten macro metrics and per-class P/R/F1 if available
    headers = ['Model', 'Class', 'Precision', 'Recall', 'F1-Score', 'LMI']
    rows = []
    lmi = metrics.get('LMI', 0.0)
    per_class = metrics.get('per_class', {})
    if isinstance(per_class, dict):
        for cls in per_class:
            if cls in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            entry = per_class[cls]
            rows.append(['BertTextCNN', str(cls), entry.get('precision', 0), entry.get('recall', 0), entry.get('f1-score', 0), lmi])
    else:
        rows.append(['BertTextCNN', 'all', metrics.get('macro_precision', 0), metrics.get('macro_recall', 0), metrics.get('macro_f1', 0), lmi])

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)