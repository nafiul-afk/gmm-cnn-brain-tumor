"""
Visual explanation of convolutional network predictions.

Gradient weighted Class Activation Mapping produces a coarse localisation map
by weighting the feature maps of the final convolutional layer with the
gradient of the predicted class score. The resulting heatmap indicates which
image regions most influenced the decision.

Interpretability is not optional in a clinical context. A model that reaches
high accuracy by attending to scanner artefacts rather than to anatomy would be
unsafe, and the heatmap is the primary evidence used in this project to argue
that the learned features are anatomically plausible.

Course mapping: CO6 (translating predictions into actionable decisions),
Lecture 08 (backpropagation).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

from . import config


def find_last_convolutional_layer(model: models.Model) -> str:
    """Return the name of the deepest convolutional layer in a model.

    Nested models such as a transfer learning backbone are searched
    recursively, because the convolutional layers of those models are not
    exposed at the top level.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, models.Model):
            try:
                inner_name = find_last_convolutional_layer(layer)
            except ValueError:
                continue
            return f"{layer.name}/{inner_name}"

    raise ValueError("The supplied model does not contain a convolutional layer.")


def _resolve_layer(model: models.Model, layer_name: str):
    """Fetch a layer that may be nested inside a sub model."""
    if "/" in layer_name:
        outer_name, inner_name = layer_name.split("/", 1)
        return model.get_layer(outer_name).get_layer(inner_name)
    return model.get_layer(layer_name)


def _build_gradient_model(model: models.Model, target_layer) -> models.Model:
    """Construct a model exposing both the feature maps and the predictions.

    Keras 2 and Keras 3 differ in how intermediate tensors of a loaded model may
    be reused. The functional construction is attempted first because it is the
    cheapest, and a rebuild from the input tensor is used as a fallback when the
    graph of a deserialised model is not directly addressable.
    """
    try:
        # Keras 3 warns when a single input model is declared with a list, then
        # called with a bare tensor, so the input is unwrapped where possible.
        model_inputs = model.inputs
        if isinstance(model_inputs, (list, tuple)) and len(model_inputs) == 1:
            model_inputs = model_inputs[0]

        return models.Model(
            inputs=model_inputs,
            outputs=[target_layer.output, model.output],
        )
    except (AttributeError, ValueError, TypeError):
        inputs = tf.keras.Input(shape=model.input_shape[1:])
        features = None
        tensor = inputs
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            tensor = layer(tensor)
            if layer is target_layer:
                features = tensor
        if features is None:
            raise ValueError(
                "The requested layer could not be located while rebuilding the "
                "gradient model."
            )
        return models.Model(inputs=inputs, outputs=[features, tensor])


def compute_gradcam_heatmap(
    model: models.Model,
    image: np.ndarray,
    class_index: Optional[int] = None,
    layer_name: Optional[str] = None,
) -> Tuple[np.ndarray, int, float]:
    """Compute a normalised Grad-CAM heatmap for a single image.

    Parameters
    ----------
    model:
        A trained classification model.
    image:
        Preprocessed image of shape ``(height, width, channels)`` with values
        in the unit interval.
    class_index:
        Class whose score is differentiated. Defaults to the predicted class.
    layer_name:
        Convolutional layer to visualise. Defaults to the deepest one.

    Returns
    -------
    tuple
        ``(heatmap, class_index, confidence)`` where the heatmap is a two
        dimensional array scaled to the unit interval.
    """
    if layer_name is None:
        layer_name = find_last_convolutional_layer(model)

    target_layer = _resolve_layer(model, layer_name)
    batch = np.expand_dims(image, axis=0).astype(np.float32)

    gradient_model = _build_gradient_model(model, target_layer)

    with tf.GradientTape() as tape:
        feature_maps, predictions = gradient_model(batch, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, feature_maps)

    if gradients is None:
        return np.zeros(image.shape[:2], dtype=np.float32), int(class_index), 0.0

    # Average the gradient over the spatial dimensions to obtain one importance
    # weight per feature channel.
    channel_weights = tf.reduce_mean(gradients, axis=(0, 1, 2))

    weighted_maps = feature_maps[0] @ channel_weights[..., tf.newaxis]
    heatmap = tf.squeeze(weighted_maps)

    # Only positive contributions towards the class of interest are retained.
    heatmap = tf.nn.relu(heatmap)
    maximum = tf.reduce_max(heatmap)
    heatmap = heatmap / (maximum + tf.keras.backend.epsilon())

    confidence = float(predictions[0][class_index])
    return heatmap.numpy(), int(class_index), confidence


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a heatmap over the original image for display."""
    base = image
    if base.dtype != np.uint8:
        base = np.clip(base * 255.0, 0, 255).astype(np.uint8)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    elif base.shape[-1] == 1:
        base = cv2.cvtColor(base[..., 0], cv2.COLOR_GRAY2RGB)

    resized = cv2.resize(heatmap, (base.shape[1], base.shape[0]))
    coloured = cv2.applyColorMap(np.uint8(255 * resized), colormap)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(base, 1.0 - alpha, coloured, alpha, 0)


def generate_explanation(
    model: models.Model,
    image: np.ndarray,
    class_index: Optional[int] = None,
    alpha: float = 0.4,
) -> dict:
    """Produce the heatmap, the overlay and the prediction in one call."""
    heatmap, resolved_index, confidence = compute_gradcam_heatmap(model, image, class_index)
    overlay = overlay_heatmap(image, heatmap, alpha)

    return {
        "heatmap": heatmap,
        "overlay": overlay,
        "class_index": resolved_index,
        "class_label": config.CLASS_LABELS[resolved_index],
        "confidence": confidence,
    }


def heatmap_mask_agreement(heatmap: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> float:
    """Measure the overlap between the Grad-CAM focus and the GMM mask.

    The value is the intersection over union of the thresholded heatmap and the
    segmentation mask. A high value provides quantitative evidence that the
    network attends to the same region that the unsupervised stage identified,
    which supports the central claim of the hybrid design.
    """
    if not mask.any():
        return 0.0

    resized_heatmap = cv2.resize(heatmap, (mask.shape[1], mask.shape[0]))
    heatmap_binary = resized_heatmap >= threshold
    mask_binary = mask > 0

    intersection = np.logical_and(heatmap_binary, mask_binary).sum()
    union = np.logical_or(heatmap_binary, mask_binary).sum()

    return float(intersection / union) if union else 0.0
