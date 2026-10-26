"""
Image preprocessing and augmentation.

Every model in this project consumes images that have passed through the same
deterministic preprocessing chain, which guarantees that performance
differences between the baseline and the hybrid pipeline are attributable to
the segmentation stage rather than to inconsistent inputs.

Course mapping: CO4 (derived attributes and transformations).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from . import config


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB or single channel image to a two dimensional array."""
    if image.ndim == 2:
        return image
    if image.shape[-1] == 1:
        return image[..., 0]
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Return an image as ``uint8`` regardless of whether it was normalised."""
    if image.dtype == np.uint8:
        return image
    maximum = float(np.nanmax(image)) if image.size else 0.0
    scaled = image * 255.0 if maximum <= 1.0 else image
    return np.clip(scaled, 0, 255).astype(np.uint8)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalisation.

    Magnetic resonance images frequently exhibit low local contrast between a
    lesion and the surrounding parenchyma. CLAHE equalises the histogram within
    small tiles and clips the amplification, which sharpens lesion boundaries
    without amplifying noise across the whole image.

    Parameters
    ----------
    image:
        Grayscale or RGB image. RGB input is enhanced on the luminance channel
        so that colour relationships are preserved.

    Returns
    -------
    numpy.ndarray
        Enhanced image with the same shape and ``uint8`` dtype as the input.
    """
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )

    source = to_uint8(image)

    if source.ndim == 2:
        return clahe.apply(source)

    if source.shape[-1] == 1:
        return clahe.apply(source[..., 0])[..., np.newaxis]

    lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def denoise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Suppress salt and pepper noise using a median filter."""
    return cv2.medianBlur(to_uint8(image), kernel_size)


def resize(image: np.ndarray, image_size: int = config.IMAGE_SIZE) -> np.ndarray:
    """Resize an image to a square of ``image_size`` pixels."""
    interpolation = cv2.INTER_AREA if image.shape[0] > image_size else cv2.INTER_LINEAR
    return cv2.resize(image, (image_size, image_size), interpolation=interpolation)


def normalise(image: np.ndarray) -> np.ndarray:
    """Scale pixel intensities into the closed interval zero to one."""
    return image.astype(np.float32) / 255.0


def preprocess_image(
    image: np.ndarray,
    image_size: int = config.IMAGE_SIZE,
    use_clahe: bool = True,
    use_denoise: bool = False,
) -> np.ndarray:
    """Run the full preprocessing chain on a single image.

    The chain is resize, optional median denoise, optional CLAHE, conversion to
    three channels and normalisation to the unit interval.
    """
    processed = to_uint8(image)

    if processed.ndim == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    elif processed.shape[-1] == 1:
        processed = cv2.cvtColor(processed[..., 0], cv2.COLOR_GRAY2RGB)
    elif processed.shape[-1] == 4:
        processed = cv2.cvtColor(processed, cv2.COLOR_RGBA2RGB)

    processed = resize(processed, image_size)

    if use_denoise:
        processed = denoise(processed)

    if use_clahe:
        processed = apply_clahe(processed)

    return normalise(processed)


def preprocess_batch(
    images: np.ndarray,
    image_size: int = config.IMAGE_SIZE,
    use_clahe: bool = True,
    use_denoise: bool = False,
) -> np.ndarray:
    """Apply :func:`preprocess_image` to an array of images."""
    return np.stack(
        [preprocess_image(image, image_size, use_clahe, use_denoise) for image in images]
    )


def load_and_preprocess(
    image_path: Path | str,
    image_size: int = config.IMAGE_SIZE,
    use_clahe: bool = True,
) -> np.ndarray:
    """Read an image from disk and return a model ready array.

    The returned array has shape ``(1, size, size, 3)`` so that it can be passed
    straight to ``model.predict``.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read an image from '{image_path}'.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed = preprocess_image(image, image_size, use_clahe)
    return np.expand_dims(processed, axis=0)


def build_augmenter(seed: int = config.RANDOM_SEED):
    """Create a Keras image generator configured for MRI augmentation.

    Vertical flips and large rotations are deliberately avoided because brain
    anatomy has a consistent superior to inferior orientation. Only
    transformations that a radiographer could plausibly produce are used.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    return ImageDataGenerator(
        rotation_range=config.AUGMENTATION["rotation_range"],
        width_shift_range=config.AUGMENTATION["width_shift_range"],
        height_shift_range=config.AUGMENTATION["height_shift_range"],
        zoom_range=config.AUGMENTATION["zoom_range"],
        horizontal_flip=config.AUGMENTATION["horizontal_flip"],
        fill_mode=config.AUGMENTATION["fill_mode"],
    )


def make_training_generator(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.RANDOM_SEED,
):
    """Return an iterator that yields augmented training batches."""
    augmenter = build_augmenter(seed)
    return augmenter.flow(x_train, y_train, batch_size=batch_size, seed=seed)


def summarise_intensity(images: np.ndarray) -> Tuple[float, float]:
    """Return the mean and standard deviation of pixel intensity."""
    flattened = images.reshape(-1)
    return float(np.mean(flattened)), float(np.std(flattened))
