"""
Unsupervised lesion localisation with Gaussian Mixture Models.

A Gaussian Mixture Model treats the intensity histogram of a magnetic
resonance slice as a weighted sum of Gaussian densities,

    p(x) = sum over k of pi_k * N(x | mu_k, sigma_k)

The parameters pi_k, mu_k and sigma_k are estimated with the Expectation
Maximisation algorithm. Because brain tissue classes occupy broadly separable
intensity bands, the fitted components approximate background, normal
parenchyma and hyperintense tissue. The hyperintense component is used as a
lesion candidate and is converted into a soft attention mask that is supplied
to the convolutional network.

Course mapping: CO7 (pattern discovery), Lecture 06 (Gaussian Mixture Models
and Expectation Maximisation), course content item on Principal Component
Analysis.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm

from . import config
from .preprocessing import apply_clahe, to_grayscale, to_uint8


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------


def fit_gmm_on_image(
    image: np.ndarray,
    n_components: int = config.GMM_N_COMPONENTS,
    random_state: int = config.RANDOM_SEED,
) -> Tuple[np.ndarray, GaussianMixture, np.ndarray]:
    """Fit a Gaussian Mixture Model to the intensity histogram of one image.

    Parameters
    ----------
    image:
        Grayscale or RGB image in any numeric range.
    n_components:
        Number of Gaussian components in the mixture.
    random_state:
        Seed passed to scikit-learn for reproducible initialisation.

    Returns
    -------
    tuple
        ``(label_map, gmm, enhanced)`` where ``label_map`` assigns a component
        index to every pixel, ``gmm`` is the fitted estimator and ``enhanced``
        is the CLAHE enhanced grayscale image the model was fitted on.
    """
    grayscale = to_grayscale(to_uint8(image))
    enhanced = apply_clahe(grayscale)

    observations = enhanced.reshape(-1, 1).astype(np.float64)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=config.GMM_COVARIANCE_TYPE,
        max_iter=config.GMM_MAX_ITER,
        random_state=random_state,
        reg_covar=1e-4,
    )
    gmm.fit(observations)

    label_map = gmm.predict(observations).reshape(enhanced.shape)
    return label_map, gmm, enhanced


def identify_lesion_component(
    label_map: np.ndarray,
    enhanced: np.ndarray,
    n_components: int = config.GMM_N_COMPONENTS,
) -> int:
    """Select the mixture component most likely to represent a lesion.

    The brightest component is often the skull or a scanner artefact rather
    than pathology, and the darkest component is background. The component is
    therefore scored by combining mean intensity with a penalty for occupying
    an implausibly large or small fraction of the slice.

    Returns
    -------
    int
        Index of the selected component.
    """
    total_pixels = float(label_map.size)
    scores: List[float] = []

    for component in range(n_components):
        mask = label_map == component
        area_ratio = float(mask.sum()) / total_pixels

        if area_ratio == 0.0:
            scores.append(-np.inf)
            continue

        mean_intensity = float(enhanced[mask].mean()) / 255.0

        # Penalise components that behave like background or like the whole
        # head rather than a focal abnormality.
        if area_ratio > config.MASK_MAX_AREA_RATIO:
            penalty = (area_ratio - config.MASK_MAX_AREA_RATIO) * 2.0
        elif area_ratio < config.MASK_MIN_AREA_RATIO:
            penalty = (config.MASK_MIN_AREA_RATIO - area_ratio) * 10.0
        else:
            penalty = 0.0

        scores.append(mean_intensity - penalty)

    return int(np.argmax(scores))


def clean_mask(mask: np.ndarray, kernel_size: int = config.MORPH_KERNEL_SIZE) -> np.ndarray:
    """Remove speckle and close small holes in a binary mask.

    An opening operation discards isolated bright pixels and a subsequent
    closing operation fills interior gaps, producing a contiguous region that
    better reflects anatomical structure.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Retain only the largest connected region of a binary mask."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    # Index zero is the background label, so the search starts at one.
    largest_index = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == largest_index).astype(np.uint8) * 255


def segment_image(
    image: np.ndarray,
    n_components: int = config.GMM_N_COMPONENTS,
    keep_largest: bool = True,
) -> Dict[str, np.ndarray]:
    """Produce a lesion mask and intermediate artefacts for one image.

    Returns
    -------
    dict
        Keys ``enhanced``, ``label_map``, ``raw_mask``, ``mask``,
        ``component_index`` and ``area_ratio``.
    """
    label_map, _, enhanced = fit_gmm_on_image(image, n_components)
    component_index = identify_lesion_component(label_map, enhanced, n_components)

    raw_mask = (label_map == component_index).astype(np.uint8) * 255
    mask = clean_mask(raw_mask)

    if keep_largest and mask.any():
        mask = largest_connected_component(mask)

    return {
        "enhanced": enhanced,
        "label_map": label_map,
        "raw_mask": raw_mask,
        "mask": mask,
        "component_index": component_index,
        "area_ratio": float(mask.sum() / 255.0) / float(mask.size),
    }


def colourise_label_map(label_map: np.ndarray, n_components: int) -> np.ndarray:
    """Render a component label map as an RGB image for visual inspection."""
    palette = np.array(
        [
            [31, 78, 121],
            [214, 137, 16],
            [30, 132, 73],
            [192, 57, 43],
            [125, 60, 152],
        ],
        dtype=np.uint8,
    )
    output = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for component in range(n_components):
        output[label_map == component] = palette[component % len(palette)]
    return output


# ---------------------------------------------------------------------------
# Attention weighting for the hybrid pipeline
# ---------------------------------------------------------------------------


def apply_attention_mask(
    image: np.ndarray,
    mask: np.ndarray,
    background_weight: float = config.ROI_BACKGROUND_WEIGHT,
    blur_kernel: int = 15,
) -> np.ndarray:
    """Attenuate the image outside the lesion candidate region.

    Hard masking discards anatomical context that the network needs in order to
    distinguish a meningioma at the brain margin from a pituitary lesion at the
    skull base. Instead the mask is blurred into a soft attention map and the
    background is retained at reduced intensity, so the network is biased
    towards the region of interest without losing global structure.

    Parameters
    ----------
    image:
        Normalised RGB image with values in the unit interval.
    mask:
        Binary mask with values of zero or two hundred and fifty five.
    background_weight:
        Intensity multiplier applied far from the mask. Zero would remove the
        background completely.
    blur_kernel:
        Size of the Gaussian kernel used to soften the mask edge.

    Returns
    -------
    numpy.ndarray
        Weighted image with the same shape and dtype as the input.
    """
    if not mask.any():
        return image

    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (blur_kernel, blur_kernel), 0)
    soft = np.clip(soft, 0.0, 1.0)

    weight = background_weight + (1.0 - background_weight) * soft

    if image.ndim == 3:
        weight = weight[..., np.newaxis]

    return (image * weight).astype(image.dtype)


def transform_for_hybrid(
    image: np.ndarray,
    n_components: int = config.GMM_N_COMPONENTS,
    background_weight: float = config.ROI_BACKGROUND_WEIGHT,
) -> np.ndarray:
    """Apply the complete hybrid transformation to one preprocessed image.

    The input is expected to already be normalised to the unit interval by
    :func:`src.preprocessing.preprocess_image`.
    """
    segmentation = segment_image(image, n_components)
    return apply_attention_mask(image, segmentation["mask"], background_weight)


def transform_batch_for_hybrid(
    images: np.ndarray,
    n_components: int = config.GMM_N_COMPONENTS,
    background_weight: float = config.ROI_BACKGROUND_WEIGHT,
    show_progress: bool = True,
    description: str = "GMM segmentation",
) -> np.ndarray:
    """Apply the hybrid transformation to an array of preprocessed images.

    Fitting one mixture model per image is the dominant cost of the hybrid
    pipeline. Results should be cached with :func:`numpy.save` so that the
    transformation is not repeated across training runs.
    """
    iterator = images
    if show_progress:
        iterator = tqdm(images, desc=description, leave=False)

    return np.stack(
        [transform_for_hybrid(image, n_components, background_weight) for image in iterator]
    )


# ---------------------------------------------------------------------------
# Feature extraction for the classical baseline
# ---------------------------------------------------------------------------


def extract_gmm_features(image: np.ndarray, n_components: int = config.GMM_N_COMPONENTS) -> np.ndarray:
    """Derive a compact descriptor from the fitted mixture and lesion mask.

    The descriptor concatenates the sorted component means, the component
    weights, the component variances and shape statistics of the lesion mask.
    It allows a classical classifier to be trained without any convolutional
    feature learning, which provides an interpretable comparison point.
    """
    label_map, gmm, enhanced = fit_gmm_on_image(image, n_components)
    component_index = identify_lesion_component(label_map, enhanced, n_components)
    mask = clean_mask((label_map == component_index).astype(np.uint8) * 255)

    order = np.argsort(gmm.means_.ravel())
    means = gmm.means_.ravel()[order] / 255.0
    weights = gmm.weights_[order]
    variances = np.sqrt(gmm.covariances_.reshape(n_components)[order]) / 255.0

    mask_binary = mask > 0
    area_ratio = float(mask_binary.sum()) / float(mask_binary.size)

    if mask_binary.any():
        rows, cols = np.nonzero(mask_binary)
        centroid_row = rows.mean() / mask.shape[0]
        centroid_col = cols.mean() / mask.shape[1]
        spread_row = rows.std() / mask.shape[0]
        spread_col = cols.std() / mask.shape[1]
        mean_intensity = float(enhanced[mask_binary].mean()) / 255.0
    else:
        centroid_row = centroid_col = 0.5
        spread_row = spread_col = 0.0
        mean_intensity = 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, True)
        area = cv2.contourArea(largest)
        circularity = (4.0 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
    else:
        circularity = 0.0

    shape_features = np.array(
        [
            area_ratio,
            centroid_row,
            centroid_col,
            spread_row,
            spread_col,
            mean_intensity,
            circularity,
        ],
        dtype=np.float64,
    )

    return np.concatenate([means, weights, variances, shape_features])


def extract_gmm_features_batch(
    images: np.ndarray,
    n_components: int = config.GMM_N_COMPONENTS,
    show_progress: bool = True,
) -> np.ndarray:
    """Compute :func:`extract_gmm_features` for an array of images."""
    iterator = images
    if show_progress:
        iterator = tqdm(images, desc="GMM feature extraction", leave=False)
    return np.stack([extract_gmm_features(image, n_components) for image in iterator])


# ---------------------------------------------------------------------------
# Principal Component Analysis
# ---------------------------------------------------------------------------


def flatten_for_pca(images: np.ndarray, image_size: int = config.PCA_IMAGE_SIZE) -> np.ndarray:
    """Downscale images to grayscale vectors suitable for PCA.

    Full resolution colour images produce a design matrix with more than one
    hundred and fifty thousand columns, which is computationally wasteful.
    Downscaling to sixty four by sixty four retains the coarse structure that
    PCA is able to model.
    """
    vectors = []
    for image in images:
        grayscale = to_grayscale(to_uint8(image))
        resized = cv2.resize(grayscale, (image_size, image_size), interpolation=cv2.INTER_AREA)
        vectors.append(resized.reshape(-1).astype(np.float32) / 255.0)
    return np.stack(vectors)


def fit_pca(
    images: np.ndarray,
    n_components: int = config.PCA_N_COMPONENTS,
    image_size: int = config.PCA_IMAGE_SIZE,
) -> Tuple[PCA, np.ndarray]:
    """Fit PCA on flattened images and return the estimator with the scores."""
    matrix = flatten_for_pca(images, image_size)
    pca = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
    scores = pca.fit_transform(matrix)
    return pca, scores


def transform_pca(
    pca: PCA, images: np.ndarray, image_size: int = config.PCA_IMAGE_SIZE
) -> np.ndarray:
    """Project new images onto a previously fitted PCA basis."""
    return pca.transform(flatten_for_pca(images, image_size))


def explained_variance_table(pca: PCA, thresholds: Sequence[float] = (0.80, 0.90, 0.95, 0.99)):
    """Report how many components are needed to reach variance thresholds."""
    import pandas as pd

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    records = []
    for threshold in thresholds:
        indices = np.where(cumulative >= threshold)[0]
        required = int(indices[0] + 1) if indices.size else None
        records.append(
            {
                "variance_threshold": threshold,
                "components_required": required,
            }
        )
    return pd.DataFrame.from_records(records)
