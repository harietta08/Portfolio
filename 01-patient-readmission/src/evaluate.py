"""
evaluate.py — Model evaluation, metrics, and threshold analysis
===============================================================
Called by train.py and notebooks/03_modeling.ipynb
Never use accuracy as headline metric — leads with AUC and clinical cost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)


COST_FALSE_NEGATIVE = 15000
COST_FALSE_POSITIVE = 500


def plot_roc_curves(models_results: dict, save_path: str = None):
    """
    Plot ROC curves for all three models on one chart.

    Args:
        models_results: {model_name: {'y_test': arr, 'y_proba': arr}}
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for i, (name, data) in enumerate(models_results.items()):
        fpr, tpr, _ = roc_curve(data['y_test'], data['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — 3 Model Comparison', fontweight='bold')
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curves(models_results: dict, save_path: str = None):
    """
    Plot Precision-Recall curves for all models.

    PR curves are more informative than ROC on imbalanced datasets.
    A model with high AUC-ROC can still have poor precision on minority class.

    Args:
        models_results: {model_name: {'y_test': arr, 'y_proba': arr}}
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline = sum(data['y_test']) / len(data['y_test']) \
        for data in list(models_results.values())[:1]

    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for i, (name, data) in enumerate(models_results.items()):
        precision, recall, _ = precision_recall_curve(data['y_test'], data['y_proba'])
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=colors[i], lw=2,
                label=f"{name} (AP = {pr_auc:.3f})")

    # Baseline: random classifier on imbalanced data
    pos_rate = list(models_results.values())[0]['y_test'].mean()
    ax.axhline(y=pos_rate, color='black', linestyle='--', lw=1,
               label=f'Random baseline ({pos_rate:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves — Imbalanced Dataset', fontweight='bold')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_threshold_analysis(y_true, y_proba, model_name: str, save_path: str = None):
    """
    Plot how precision, recall, F1, and clinical cost change with threshold.

    This is the core of cost-sensitive threshold optimization.
    Default threshold of 0.5 is rarely optimal for imbalanced clinical data.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        model_name: String for plot title
        save_path: Optional save path
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    precisions, recalls, f1s, costs = [], [], [], []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        cost = (fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        costs.append(cost)

    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Precision / Recall / F1 vs threshold
    axes[0].plot(thresholds, precisions, label='Precision', color='#3498db', lw=2)
    axes[0].plot(thresholds, recalls, label='Recall', color='#e74c3c', lw=2)
    axes[0].plot(thresholds, f1s, label='F1', color='#2ecc71', lw=2)
    axes[0].axvline(x=optimal_threshold, color='black', linestyle='--',
                    label=f'Optimal threshold ({optimal_threshold:.2f})')
    axes[0].set_xlabel('Decision Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title(f'Metrics vs Threshold — {model_name}', fontweight='bold')
    axes[0].legend()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Clinical cost vs threshold
    axes[1].plot(thresholds, [c / 1000 for c in costs],
                 color='#e67e22', lw=2)
    axes[1].axvline(x=optimal_threshold, color='black', linestyle='--',
                    label=f'Min cost at threshold={optimal_threshold:.2f}')
    axes[1].set_xlabel('Decision Threshold')
    axes[1].set_ylabel('Total Clinical Cost ($000s)')
    axes[1].set_title('Clinical Cost vs Threshold\n(FN=$15K, FP=$500)', fontweight='bold')
    axes[1].legend()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.suptitle(f'Threshold Optimization — {model_name}', fontweight='bold', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Optimal threshold: {optimal_threshold:.2f}")
    print(f"Min clinical cost: ${min(costs):,}")
    return optimal_threshold


def plot_confusion_matrix(y_true, y_proba, threshold: float,
                           model_name: str, save_path: str = None):
    """
    Plot confusion matrix with clinical cost annotations.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        threshold: Decision threshold to apply
        model_name: String for title
        save_path: Optional save path
    """
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Not Readmitted', 'Readmitted <30d']
    )
    disp.plot(ax=ax, colorbar=False, cmap='Blues')

    # Annotate with clinical cost
    ax.set_title(
        f'Confusion Matrix — {model_name} (threshold={threshold})\n'
        f'FN cost: {fn} × $15,000 = ${fn*15000:,} | '
        f'FP cost: {fp} × $500 = ${fp*500:,}',
        fontweight='bold', fontsize=10
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"True Positives  (caught high-risk):      {tp:,}")
    print(f"False Negatives (missed high-risk):      {fn:,}  → ${fn*COST_FALSE_NEGATIVE:,} cost")
    print(f"False Positives (unnecessary follow-up): {fp:,}  → ${fp*COST_FALSE_POSITIVE:,} cost")
    print(f"True Negatives  (correctly cleared):     {tn:,}")


def build_model_comparison_table(all_results: list) -> pd.DataFrame:
    """
    Build a formatted comparison table for README and notebook.

    Args:
        all_results: List of metric dicts from compute_metrics()

    Returns:
        Formatted DataFrame
    """
    rows = []
    for r in all_results:
        rows.append({
            'Model': r['model_name'].replace('_', ' ').title(),
            'AUC-ROC': f"{r['auc_roc']:.4f}",
            'Avg Precision': f"{r['avg_precision']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1': f"{r['f1']:.4f}",
            'Threshold': f"{r['threshold']:.2f}",
            'False Negatives': r['fn'],
            'Clinical Cost': f"${r['total_clinical_cost']:,}",
        })

    df = pd.DataFrame(rows)
    return df
