"""
Dataset discovery, loading and splitting utilities.

The Kaggle Brain Tumor MRI dataset is distributed as two top level folders,
``Training`` and ``Testing``, each containing one subfolder per class. This
module turns that folder hierarchy into NumPy arrays suitable for both the
classical machine learning pipeline and the convolutional neural network.

Course mapping: CO1 (data acquisition and integration), CO2 (data quality
inspection).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from . import config

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def verify_dataset(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Check that the expected dataset folders exist and report image counts.

    Parameters
    ----------
    data_dir:
        Root folder containing ``Training`` and ``Testing``. Defaults to the
        path configured in :mod:`src.config`.

    Returns
    -------
    pandas.DataFrame
        One row per split and class with the number of image files found.

    Raises
    ------
    FileNotFoundError
        If either split folder is missing.
    """
    root = Path(data_dir) if data_dir is not None else config.DATA_DIR
    records: List[Dict[str, object]] = []

    for split in ("Training", "Testing"):
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Expected directory '{split_dir}' was not found. Download the "
                "dataset from Kaggle and extract it into the data folder."
            )
        for class_name in config.CLASS_FOLDERS:
            class_dir = split_dir / class_name
            count = len(list_image_files(class_dir)) if class_dir.is_dir() else 0
            records.append(
                {
                    "split": split,
                    "class": class_name,
                    "label": config.CLASS_LABELS[config.LABEL_TO_INDEX[class_name]],
                    "image_count": count,
                    "exists": class_dir.is_dir(),
                }
            )

    return pd.DataFrame.from_records(records)


def list_image_files(directory: Path | str) -> List[Path]:
    """Return every image file inside ``directory`` sorted by filename."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]
    return sorted(files)


def load_split(
    split_dir: Path | str,
    image_size: int = config.IMAGE_SIZE,
    grayscale: bool = False,
    limit_per_class: int | None = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load every image from one dataset split into memory.

    Parameters
    ----------
    split_dir:
        Folder containing one subfolder per class.
    image_size:
        Target width and height in pixels.
    grayscale:
        When ``True`` images are returned with a single channel.
    limit_per_class:
        Optional cap on the number of images read per class. Useful for quick
        experiments on constrained hardware.
    show_progress:
        Display a progress bar while reading files.

    Returns
    -------
    tuple
        ``(images, labels, file_paths)`` where ``images`` has shape
        ``(n, size, size, channels)`` with dtype ``uint8`` and ``labels``
        contains integer class indices.
    """
    split_dir = Path(split_dir)
    images: List[np.ndarray] = []
    labels: List[int] = []
    file_paths: List[str] = []

    read_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR

    for class_name in config.CLASS_FOLDERS:
        class_index = config.LABEL_TO_INDEX[class_name]
        class_files = list_image_files(split_dir / class_name)
        if limit_per_class is not None:
            class_files = class_files[:limit_per_class]

        iterator = class_files
        if show_progress:
            iterator = tqdm(class_files, desc=f"{split_dir.name}/{class_name}", leave=False)

        for file_path in iterator:
            image = cv2.imread(str(file_path), read_flag)
            if image is None:
                continue
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
            if not grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
            labels.append(class_index)
            file_paths.append(str(file_path))

    if not images:
        raise RuntimeError(f"No readable images were found under '{split_dir}'.")

    image_array = np.asarray(images, dtype=np.uint8)
    if grayscale:
        image_array = image_array[..., np.newaxis]

    return image_array, np.asarray(labels, dtype=np.int64), file_paths


def load_dataset(
    data_dir: Path | str | None = None,
    image_size: int = config.IMAGE_SIZE,
    grayscale: bool = False,
    limit_per_class: int | None = None,
) -> Dict[str, object]:
    """Load both the training and testing splits.

    Returns
    -------
    dict
        Keys ``x_train``, ``y_train``, ``train_paths``, ``x_test``, ``y_test``
        and ``test_paths``.
    """
    root = Path(data_dir) if data_dir is not None else config.DATA_DIR

    x_train, y_train, train_paths = load_split(
        root / "Training", image_size, grayscale, limit_per_class
    )
    x_test, y_test, test_paths = load_split(
        root / "Testing", image_size, grayscale, limit_per_class
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "train_paths": train_paths,
        "x_test": x_test,
        "y_test": y_test,
        "test_paths": test_paths,
    }


def stratified_validation_split(
    images: np.ndarray,
    labels: np.ndarray,
    validation_size: float = config.VALIDATION_SPLIT,
    random_state: int = config.RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split training data into training and validation subsets.

    Stratification preserves the original class proportions in both subsets,
    which matters because the dataset is mildly imbalanced.
    """
    return train_test_split(
        images,
        labels,
        test_size=validation_size,
        random_state=random_state,
        stratify=labels,
    )


def class_distribution(labels: Sequence[int]) -> pd.DataFrame:
    """Summarise how many samples belong to each class."""
    labels = np.asarray(labels)
    records = []
    total = len(labels)
    for index, folder_name in enumerate(config.CLASS_FOLDERS):
        count = int(np.sum(labels == index))
        records.append(
            {
                "class_index": index,
                "class": folder_name,
                "label": config.CLASS_LABELS[index],
                "count": count,
                "percentage": round(100.0 * count / total, 2) if total else 0.0,
            }
        )
    return pd.DataFrame.from_records(records)


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Compute inverse frequency weights to offset class imbalance.

    The weights are normalised so that their mean equals one, which keeps the
    effective learning rate comparable to unweighted training.
    """
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=config.NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (config.NUM_CLASSES * counts)
    weights = weights / weights.mean()
    return {index: float(weight) for index, weight in enumerate(weights)}


def describe_image_properties(file_paths: Sequence[str], sample_size: int = 500) -> pd.DataFrame:
    """Inspect raw file dimensions before any resizing is applied.

    Reading a sample rather than the full dataset keeps the exploratory step
    fast while still revealing whether the source images share a common
    resolution.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    paths = list(file_paths)
    if len(paths) > sample_size:
        chosen_indices = rng.choice(len(paths), size=sample_size, replace=False)
        paths = [paths[i] for i in chosen_indices]

    records = []
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        records.append(
            {
                "path": path,
                "width": width,
                "height": height,
                "channels": channels,
                "aspect_ratio": round(width / height, 3),
                "file_size_kb": round(os.path.getsize(path) / 1024, 2),
            }
        )
    return pd.DataFrame.from_records(records)
