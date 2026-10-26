"""
Model evaluation, statistical testing and figure generation.

All reported numbers are produced by this module so that the notebooks, the
training script and the application remain consistent with one another.

Course mapping: CO5 (sound validation and interpretation), CO8 (comparison of
alternatives on performance metrics), Lecture 02 (evaluation metrics and cross
validation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from . import config

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
) -> Dict[str, float]:
    """Compute the headline classification metrics for one model.

    Weighted averaging is used for the summary figures because the dataset is
    mildly imbalanced. Macro averaging is reported alongside it so that
    performance on the smaller classes is not hidden.
    """
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
    }


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[str] = config.CLASS_LABELS,
) -> pd.DataFrame:
    """Return precision, recall, F1 and support for every class."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=list(class_labels),
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in class_labels:
        entry = report[label]
        rows.append(
            {
                "class": label,
                "precision": round(entry["precision"], 4),
                "recall": round(entry["recall"], 4),
                "f1_score": round(entry["f1-score"], 4),
                "support": int(entry["support"]),
            }
        )
    return pd.DataFrame.from_records(rows)


def clinical_error_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    healthy_index: int = 2,
) -> Dict[str, float]:
    """Quantify the two error types that matter most in a screening context.

    A false negative occurs when a scan containing a tumour is labelled as
    healthy. This is the most serious failure mode because it delays
    treatment. A false positive occurs when a healthy scan is flagged, which
    causes unnecessary follow up but is far less harmful.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tumour_present = y_true != healthy_index
    predicted_healthy = y_pred == healthy_index

    false_negatives = int(np.sum(tumour_present & predicted_healthy))
    total_tumour = int(np.sum(tumour_present))

    healthy_present = y_true == healthy_index
    predicted_tumour = y_pred != healthy_index
    false_positives = int(np.sum(healthy_present & predicted_tumour))
    total_healthy = int(np.sum(healthy_present))

    return {
        "false_negatives": false_negatives,
        "total_tumour_cases": total_tumour,
        "false_negative_rate": round(false_negatives / total_tumour, 4) if total_tumour else 0.0,
        "false_positives": false_positives,
        "total_healthy_cases": total_healthy,
        "false_positive_rate": round(false_positives / total_healthy, 4) if total_healthy else 0.0,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: Sequence[str] = config.CLASS_LABELS,
    title: str = "Confusion matrix",
    normalise: bool = False,
    save_path: Path | str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Render a confusion matrix as an annotated heatmap."""
    matrix = confusion_matrix(y_true, y_pred)

    if normalise:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        display_matrix = matrix / row_sums
        fmt = ".2f"
    else:
        display_matrix = matrix
        fmt = "d"

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))

    sns.heatmap(
        display_matrix,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=True,
        xticklabels=list(class_labels),
        yticklabels=list(class_labels),
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    if save_path is not None:
        ax.figure.savefig(save_path)

    return ax


def plot_training_history(
    history: Dict[str, List[float]] | object,
    title_prefix: str = "",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot accuracy and loss curves for the training and validation splits."""
    data = history.history if hasattr(history, "history") else history

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(data["accuracy"], label="Training", linewidth=2)
    if "val_accuracy" in data:
        axes[0].plot(data["val_accuracy"], label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{title_prefix}Accuracy".strip())
    axes[0].legend()

    axes[1].plot(data["loss"], label="Training", linewidth=2)
    if "val_loss" in data:
        axes[1].plot(data["val_loss"], label="Validation", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{title_prefix}Loss".strip())
    axes[1].legend()

    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path)

    return figure


def plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_labels: Sequence[str] = config.CLASS_LABELS,
    title: str = "Receiver operating characteristic",
    save_path: Path | str | None = None,
    ax: plt.Axes | None = None,
) -> Tuple[plt.Axes, Dict[str, float]]:
    """Plot one versus rest ROC curves and return the area under each curve."""
    y_binary = label_binarize(y_true, classes=list(range(len(class_labels))))

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))

    auc_scores: Dict[str, float] = {}

    for index, label in enumerate(class_labels):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_binary[:, index], y_score[:, index]
        )
        area = auc(false_positive_rate, true_positive_rate)
        auc_scores[label] = float(area)
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            color=config.CLASS_COLORS[index % len(config.CLASS_COLORS)],
            label=f"{label} (AUC = {area:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)

    auc_scores["macro_average"] = float(np.mean(list(auc_scores.values())))

    if save_path is not None:
        ax.figure.savefig(save_path)

    return ax, auc_scores


def plot_model_comparison(
    metrics_frame: pd.DataFrame,
    metrics: Sequence[str] = ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted"),
    title: str = "Model comparison",
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Draw a grouped bar chart comparing several models across metrics."""
    melted = metrics_frame.melt(
        id_vars="model",
        value_vars=list(metrics),
        var_name="metric",
        value_name="score",
    )

    figure, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="metric", y="score", hue="model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(title="Model", loc="lower right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path)

    return figure


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, float]:
    """Test whether two classifiers differ significantly on the same test set.

    McNemar's test is the appropriate paired test here because both models are
    evaluated on identical samples. It examines only the discordant pairs, that
    is the cases where exactly one of the two models is correct.

    Returns
    -------
    dict
        Counts of the discordant cells, the test statistic and the p value.
    """
    from statsmodels.stats.contingency_tables import mcnemar

    correct_a = np.asarray(y_pred_a) == np.asarray(y_true)
    correct_b = np.asarray(y_pred_b) == np.asarray(y_true)

    both_correct = int(np.sum(correct_a & correct_b))
    only_a_correct = int(np.sum(correct_a & ~correct_b))
    only_b_correct = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    table = [[both_correct, only_a_correct], [only_b_correct, both_wrong]]

    # The exact binomial form is used when the discordant count is small,
    # otherwise the chi squared approximation with continuity correction.
    use_exact = (only_a_correct + only_b_correct) < 25
    result = mcnemar(table, exact=use_exact, correction=not use_exact)

    return {
        "both_correct": both_correct,
        "only_model_a_correct": only_a_correct,
        "only_model_b_correct": only_b_correct,
        "both_incorrect": both_wrong,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "test_used": "exact binomial" if use_exact else "chi squared with continuity correction",
        "significant_at_0.05": bool(result.pvalue < 0.05),
    }


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "accuracy",
    n_iterations: int = 2000,
    confidence: float = 0.95,
    random_state: int = config.RANDOM_SEED,
) -> Dict[str, float]:
    """Estimate a confidence interval for a metric by bootstrap resampling.

    Reporting an interval rather than a single point estimate communicates the
    uncertainty that arises from a finite test set.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_samples = len(y_true)

    scorers = {
        "accuracy": lambda t, p: accuracy_score(t, p),
        "f1_weighted": lambda t, p: f1_score(t, p, average="weighted", zero_division=0),
        "f1_macro": lambda t, p: f1_score(t, p, average="macro", zero_division=0),
    }
    scorer = scorers[metric]

    scores = []
    for _ in range(n_iterations):
        indices = rng.integers(0, n_samples, size=n_samples)
        scores.append(scorer(y_true[indices], y_pred[indices]))

    lower_percentile = 100 * (1 - confidence) / 2
    upper_percentile = 100 - lower_percentile

    return {
        "metric": metric,
        "point_estimate": float(scorer(y_true, y_pred)),
        "lower_bound": float(np.percentile(scores, lower_percentile)),
        "upper_bound": float(np.percentile(scores, upper_percentile)),
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_metrics(payload: Dict | List, filename: str, directory: Path | str | None = None) -> Path:
    """Write a metrics dictionary to a JSON file in the results folder."""
    directory = Path(directory) if directory is not None else config.METRICS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_metrics(filename: str, directory: Path | str | None = None) -> Dict | List:
    """Read a metrics JSON file previously written by :func:`save_metrics`."""
    directory = Path(directory) if directory is not None else config.METRICS_DIR
    with (Path(directory) / filename).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_comparison_table(metric_dicts: Iterable[Dict[str, float]]) -> pd.DataFrame:
    """Combine several metric dictionaries into a single sorted table."""
    frame = pd.DataFrame(list(metric_dicts))
    if "f1_weighted" in frame.columns:
        frame = frame.sort_values("f1_weighted", ascending=False).reset_index(drop=True)
    return frame
