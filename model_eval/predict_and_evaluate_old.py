from datetime import datetime
from pathlib import Path
import base64

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score

from config.paths import PatientDir, PATHS
from utils.io import pickle_path, save_dataframe_multiformat


def compute_metric(values: dict, metric: str):
    def safe_div(numerator, denominator, default: float = 0.0):
        return numerator / denominator if denominator != 0 else default

    tp, tn, fp, fn = values['tp'], values['tn'], values['fp'], values['fn']
    match metric:
        case 'precision':
            return safe_div(tp, tp + fp)
        case 'recall':
            return safe_div(tp, tp + fn)
        case 'specificity':
            return safe_div(tn, tn + fp)
        case 'sensitivity':
            return safe_div(tp, tp + fn)
        case 'f1':
            pre = compute_metric(values, 'precision')
            rec = compute_metric(values, 'recall')
            return 2 * safe_div(pre * rec, pre + rec)
    raise ValueError('Unsupported metric: ' + metric)


def compute_threshold_metrics(y_true, y_pred_prob, threshold_precision=0.005, thresholds=None):
    """Compute per-threshold metrics plus ROC data for later reuse."""
    y_true = np.array(y_true).astype(int)
    y_pred_prob = np.array(y_pred_prob).flatten()

    if thresholds is None:
        # noinspection PyTypeChecker
        rounded_probs = np.round(np.round(y_pred_prob / threshold_precision) * threshold_precision, 3)
        thresholds = np.unique(rounded_probs)
    thresholds = np.asarray(thresholds)

    per_threshold = []

    for threshold in thresholds:
        preds = (y_pred_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        values = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

        per_threshold.append({
            'threshold': threshold,
            **values,
            **{metric: compute_metric(values, metric) for metric in ['precision', 'recall', 'f1', 'specificity']},
        })

    per_threshold = DataFrame(per_threshold)
    per_threshold.set_index('threshold', inplace=True, drop=True)

    # todo does it make sense to have this split up (roc vs. other metrics)?
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    return {
        'y_true': y_true,
        # todo this should be called y_pred_prob or be removed all together
        'y_pred': y_pred_prob,
        'per_threshold': per_threshold,
        'roc': {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': roc_thresholds,
            'auc': roc_auc,
        },
    }


# todo compute threshold metrics has this too (combine)
def _confusion_summary(tn, fp, fn, tp):
    denom = 2 * tp + fp + fn
    return {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1': (2 * tp / denom) if denom > 0 else 0,
    }


def select_best_threshold(metrics_bundle, metric='f1'):
    """Pick the optimal threshold for a metric from cached threshold metrics."""
    y_true = metrics_bundle['y_true']
    y_pred = metrics_bundle['y_pred']

    if metric == 'f1':
        per_thr: DataFrame = metrics_bundle['per_threshold']
        # todo this is stupid. what is this syntax?!
        scores: Series = per_thr['f1']
        thresh = scores.idxmax()
        score = scores.loc[thresh]
        summary = {
            'tp': int(per_thr['tp'][idx]),
            'tn': int(per_thr['tn'][idx]),
            'fp': int(per_thr['fp'][idx]),
            'fn': int(per_thr['fn'][idx]),
            'sensitivity': per_thr['recall'][idx],
            'specificity': per_thr['specificity'][idx],
            'precision': per_thr['precision'][idx],
            'recall': per_thr['recall'][idx],
            'f1': per_thr['f1'][idx],
        }
    elif metric == 'roc':
        roc = metrics_bundle['roc']
        j_scores = roc['tpr'] - roc['fpr']
        idx = int(np.argmax(j_scores))
        threshold = roc['thresholds'][idx]
        score = j_scores[idx]
        preds = (y_pred >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        summary = _confusion_summary(tn, fp, fn, tp)
        summary['f1'] = f1_score(y_true, preds, zero_division=0)
    else:
        raise ValueError(f"Unsupported metric '{metric}'")

    return threshold, score, summary


def plot_threshold_metrics(metrics_bundle, output_path=None):
    """Plot the cached threshold metrics and ROC curve."""
    per_thr = metrics_bundle['per_threshold']
    thresholds = per_thr['threshold']

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(thresholds, per_thr['precision'], label='Precision', linewidth=2)
    ax.plot(thresholds, per_thr['recall'], label='Recall', linewidth=2)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision vs Recall')
    ax.legend()

    ax = axes[0, 1]
    ax.plot(thresholds, per_thr['f1'], label='F1 Score', linewidth=2, color='green')
    ax.axvline(thresholds[np.argmax(per_thr['f1'])], color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Across Thresholds')

    ax = axes[1, 0]
    ax.plot(thresholds, per_thr['specificity'], label='Specificity', linewidth=2)
    ax.plot(thresholds, per_thr['recall'], label='Sensitivity (Recall)', linewidth=2)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Sensitivity vs Specificity')
    ax.legend()

    roc = metrics_bundle['roc']
    ax = axes[1, 1]
    ax.plot(roc['fpr'], roc['tpr'], label=f"ROC Curve (AUC = {roc['auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()

    for axis in axes.flatten():
        axis.grid(True, alpha=0.3)
        axis.tick_params(labelbottom=True, labelleft=True)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.close(fig)


def _ensure_results_dir(pdir: PatientDir, model: str, data_split: str) -> Path:
    """Return pdir.model_eval_dir/<model>/<split>, creating it if necessary."""
    results_dir = pdir.model_eval_dir / model / data_split
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _generate_html_report(
        pdir: PatientDir,
        model: str,
        data_split: str,
        y_true: np.ndarray,
        best_thresholds: dict,
        plot_path: Path,
) -> Path:
    """Write an HTML summary for a given model/split evaluation and return its path."""
    results_dir = _ensure_results_dir(pdir, model, data_split)
    html_path = results_dir / 'report.html'

    best_thresh, _, f1_metrics = best_thresholds['f1']

    if plot_path.exists():
        encoded = base64.b64encode(plot_path.read_bytes()).decode('ascii')
        plot_section = f'<img src="data:image/png;base64,{encoded}" alt="Metrics Plot">'
    else:
        plot_section = f'<p><em>Plot not available ({plot_path.name})</em></p>'

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Evaluation Report - {model} ({data_split.upper()})</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 30px; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 30px; }}
            h2 {{ color: #34495e; margin-top: 30px; margin-bottom: 15px; border-left: 4px solid #3498db; padding-left: 10px; }}
            .info-box {{ background-color: #ecf0f1; border-left: 4px solid #3498db; padding: 15px; margin: 15px 0; border-radius: 4px; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin: 20px 0; }}
            .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #fff; padding: 18px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.95em; }}
            th {{ background-color: #3498db; color: #fff; padding: 12px; text-align: left; }}
            td {{ padding: 12px; border-bottom: 1px solid #ecf0f1; }}
            tr:hover {{ background-color: #f9f9f9; }}
            .image-container {{ text-align: center; margin: 30px 0; }}
            .image-container img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .timestamp {{ color: #7f8c8d; font-size: 0.9em; margin-top: 20px; padding-top: 20px; border-top: 1px solid #ecf0f1; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Evaluation Report: {model} ({data_split.upper()})</h1>
            <div class="info-box">
                <strong>Patient:</strong> {pdir.name}<br>
                <strong>Data Split:</strong> {data_split.upper()}<br>
                <strong>Total Clips Evaluated:</strong> {len(y_true)}<br>
                <strong>Preictal Clips:</strong> {y_true.sum()} ({100 * y_true.sum() / len(y_true):.1f}%)<br>
                <strong>Non-Preictal Clips:</strong> {len(y_true) - y_true.sum()} ({100 * (len(y_true) - y_true.sum()) / len(y_true):.1f}%)
            </div>
            <h2>Key Metrics (F1-Optimized Threshold: {best_thresh:.4f})</h2>
            <div class="metrics-grid">
                <div class="metric-card"><div>Sensitivity (TPR)</div><div style="font-size:2em;">{f1_metrics['sensitivity']:.3f}</div></div>
                <div class="metric-card"><div>Specificity (TNR)</div><div style="font-size:2em;">{f1_metrics['specificity']:.3f}</div></div>
                <div class="metric-card"><div>Precision</div><div style="font-size:2em;">{f1_metrics['precision']:.3f}</div></div>
                <div class="metric-card"><div>Recall</div><div style="font-size:2em;">{f1_metrics['recall']:.3f}</div></div>
                <div class="metric-card"><div>F1 Score</div><div style="font-size:2em;">{f1_metrics['f1']:.3f}</div></div>
            </div>
            <h2>Confusion Matrix (F1-Optimized)</h2>
            <table>
                <tr><th></th><th>Predicted Negative</th><th>Predicted Positive</th></tr>
                <tr><td><strong>Actual Negative</strong></td><td>{f1_metrics['tn']} (TN)</td><td>{f1_metrics['fp']} (FP)</td></tr>
                <tr><td><strong>Actual Positive</strong></td><td>{f1_metrics['fn']} (FN)</td><td>{f1_metrics['tp']} (TP)</td></tr>
            </table>
            <h2>Threshold Comparison</h2>
            <table>
                <thead>
                    <tr><th>Metric</th><th>Best Threshold</th><th>Score</th><th>Sensitivity</th><th>Specificity</th><th>Precision</th><th>F1 Score</th></tr>
                </thead>
                <tbody>
    """

    for metric_name, (thresh, score, metrics) in best_thresholds.items():
        html_content += (
            f"<tr><td><strong>{metric_name.upper()}</strong></td>"
            f"<td>{thresh:.4f}</td><td>{score:.4f}</td>"
            f"<td>{metrics['sensitivity']:.4f}</td><td>{metrics['specificity']:.4f}</td>"
            f"<td>{metrics['precision']:.4f}</td><td>{metrics['f1']:.4f}</td></tr>"
        )

    html_content += f"""
                </tbody>
            </table>
            <h2>Performance Visualizations</h2>
            <div class="image-container">
                {plot_section}
                <p><em>Threshold Performance Analysis and ROC Curve</em></p>
            </div>
            <div class="timestamp">Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
    </body>
    </html>
    """

    html_path.write_text(html_content)
    return html_path


def predict_and_eval_ptnt(
        pdir: PatientDir,
        splits: tuple[str] = ('train', 'test'),
        models: tuple[str] = ('CNN', 'ensemble'),
):
    """Evaluate both train and test splits for every model and persist results."""
    clips = pd.read_pickle(pickle_path(pdir.clips_table))
    clips = clips[clips['valid']]
    split_idx = pd.read_pickle(pickle_path(pdir.train_test_split))['segment_index']

    split_masks = {
        'train': clips['end_seg'] <= split_idx,
        'test': clips['end_seg'] > split_idx,
    }

    for split_name in splits:
        split_clips = clips[split_masks[split_name]]
        if split_clips.empty:
            print(f"[WARN] No clips available for patient {pdir.name} in {split_name} split.")
            continue

        y_true = split_clips['preictal'].values

        for model in models:
            prob_column = f'{model}_probability'
            if prob_column not in split_clips:
                print(f"[WARN] Missing column {prob_column} for patient {pdir.name}; skipping.")
                continue

            probabilities = split_clips[prob_column].values
            results_dir = _ensure_results_dir(pdir, model, split_name)
            plot_path = results_dir / 'metrics_plot.png'

            metrics_bundle = compute_threshold_metrics(y_true, probabilities)
            plot_threshold_metrics(metrics_bundle, output_path=plot_path)

            best_thresholds = {}
            for metric in ['f1', 'roc']:
                best_thresh, best_score, metrics = select_best_threshold(metrics_bundle, metric=metric)
                best_thresholds[metric] = (best_thresh, best_score, metrics)

            f1_thresh, _, f1_metrics = best_thresholds['f1']
            metrics_to_save = {
                'Model': model,
                'Data_Split': split_name,
                'Total_Clips': len(y_true),
                'Preictal_Clips': int(y_true.sum()),
                'Non_Preictal_Clips': int(len(y_true) - y_true.sum()),
                'F1_Threshold': f1_thresh,
                'F1_Score': f1_metrics['f1'],
                'Sensitivity': f1_metrics['sensitivity'],
                'Specificity': f1_metrics['specificity'],
                'Precision': f1_metrics['precision'],
                'Recall': f1_metrics['recall'],
                'TP': f1_metrics['tp'],
                'TN': f1_metrics['tn'],
                'FP': f1_metrics['fp'],
                'FN': f1_metrics['fn'],
                'ROC_AUC': metrics_bundle['roc']['auc'],
            }

            metrics_table_path = results_dir / 'metrics.csv'
            save_dataframe_multiformat(pd.Series(metrics_to_save), metrics_table_path)

            threshold_csv_path = results_dir / 'threshold_analysis.csv'
            save_dataframe_multiformat(
                pd.DataFrame({
                    'threshold': per_thr['thresholds'],
                    'precision': per_thr['precision'],
                    'recall': per_thr['recall'],
                    'f1': per_thr['f1'],
                    'specificity': per_thr['specificity'],
                    'tp': per_thr['tp'],
                    'tn': per_thr['tn'],
                    'fp': per_thr['fp'],
                    'fn': per_thr['fn'],
                }),
                threshold_csv_path
            )

            _generate_html_report(
                pdir,
                model,
                split_name,
                y_true,
                best_thresholds,
                plot_path=plot_path,
            )

            print(f"✓ Saved evaluation results for patient [{pdir.name}], model [{model}], split [{split_name}]")


def main(pdirs: list[PatientDir] = PATHS.patient_dirs()):
    """Evaluate every provided patient directory."""
    for pdir in pdirs:
        print(f"======== Processing patient: {pdir.name}")
        predict_and_eval_ptnt(pdir)


if __name__ == "__main__":
    main([
        PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-01-MINIFAKE'),
        # PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-1'),
        # PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-2'),
        # PatientDir('/data/home/webb/UNEEG/datasets/competition/competition-3'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-01'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-01-FAKE'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-03'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-04'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-05'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-07'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-12'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-15'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-16'),
        # PatientDir('/data/home/webb/UNEEG/datasets/ultra2/U002-DE01-17')
    ])
