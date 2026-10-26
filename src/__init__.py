"""
Brain tumour classification from magnetic resonance imaging.

A hybrid pipeline that combines unsupervised Gaussian Mixture Model
segmentation with supervised convolutional classification.

Developed for DS 4889 Machine Learning, Department of Computer Science and
Engineering, United International University.
"""

__version__ = "1.0.0"

from . import config  # noqa: F401

__all__ = [
    "config",
    "data_loader",
    "preprocessing",
    "gmm_segmentation",
    "cnn_model",
    "classical_models",
    "evaluate",
    "explainability",
]
