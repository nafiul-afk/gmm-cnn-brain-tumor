"""
Classical machine learning baselines.

These models operate on engineered descriptors rather than on raw pixels. They
establish how much of the classification problem can be solved without
representation learning, which is the reference point against which the
convolutional results are interpreted.

Course mapping: Lecture 03 (logistic regression and decision boundaries),
Lecture 05 (maximum likelihood estimation), course content items on naive
Bayes, random forests and ensemble methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config


def build_logistic_regression() -> Pipeline:
    """Multinomial logistic regression on standardised features."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                # scikit-learn 1.7 removed the multi_class argument. The lbfgs
                # solver now applies the multinomial formulation by default when
                # more than two classes are present.
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def build_gaussian_naive_bayes() -> Pipeline:
    """Gaussian naive Bayes classifier.

    The model assumes conditional independence between features given the
    class, and estimates each class conditional density by maximum likelihood.
    It is the probabilistic reference model described in the course text.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB()),
        ]
    )


def build_random_forest(n_estimators: int = 300) -> Pipeline:
    """Random forest classifier, an instance of bagging over decision trees."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def build_support_vector_machine() -> Pipeline:
    """Support vector classifier with a radial basis function kernel."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    random_state=config.RANDOM_SEED,
                ),
            ),
        ]
    )


def build_stacking_ensemble() -> StackingClassifier:
    """Stack the classical models under a logistic regression meta learner.

    Stacking learns how much to trust each base estimator rather than giving
    them equal votes, which typically outperforms simple majority voting when
    the base learners have differing error profiles.
    """
    estimators = [
        ("logistic", build_logistic_regression()),
        ("naive_bayes", build_gaussian_naive_bayes()),
        ("random_forest", build_random_forest()),
        ("svm", build_support_vector_machine()),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=config.RANDOM_SEED),
        cv=5,
        n_jobs=-1,
    )


MODEL_BUILDERS = {
    "logistic_regression": build_logistic_regression,
    "naive_bayes": build_gaussian_naive_bayes,
    "random_forest": build_random_forest,
    "svm": build_support_vector_machine,
    "stacking_ensemble": build_stacking_ensemble,
}


@dataclass
class ClassicalBaselineResult:
    """Container holding a fitted estimator together with its test scores."""

    name: str
    estimator: object
    accuracy: float
    f1_weighted: float
    predictions: np.ndarray = field(repr=False)
    probabilities: np.ndarray = field(repr=False)


def train_classical_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_names: Tuple[str, ...] = tuple(MODEL_BUILDERS),
) -> Dict[str, ClassicalBaselineResult]:
    """Fit and evaluate each classical model on engineered features.

    Parameters
    ----------
    x_train, x_test:
        Two dimensional feature matrices.
    y_train, y_test:
        Integer class labels.
    model_names:
        Subset of :data:`MODEL_BUILDERS` keys to train.

    Returns
    -------
    dict
        Mapping from model name to :class:`ClassicalBaselineResult`.
    """
    from sklearn.metrics import accuracy_score, f1_score

    results: Dict[str, ClassicalBaselineResult] = {}

    for name in model_names:
        estimator = MODEL_BUILDERS[name]()
        estimator.fit(x_train, y_train)

        predictions = estimator.predict(x_test)
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(x_test)
        else:
            probabilities = np.zeros((len(predictions), config.NUM_CLASSES))

        results[name] = ClassicalBaselineResult(
            name=name,
            estimator=estimator,
            accuracy=float(accuracy_score(y_test, predictions)),
            f1_weighted=float(f1_score(y_test, predictions, average="weighted")),
            predictions=predictions,
            probabilities=probabilities,
        )

    return results


def save_estimator(estimator: object, path: Path | str) -> None:
    """Persist a fitted scikit-learn estimator to disk."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path)


def load_estimator(path: Path | str):
    """Load a previously persisted scikit-learn estimator."""
    import joblib

    return joblib.load(Path(path))
