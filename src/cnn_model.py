"""
Convolutional neural network architectures.

The same architecture is used for the baseline and for the hybrid experiment so
that any difference in test performance can be attributed to the Gaussian
Mixture Model segmentation stage rather than to model capacity.

Course mapping: Lectures 08 and 09 (neural network fundamentals, activation
functions, backpropagation, batch normalisation, dropout, optimisers) and
Lecture 11 (convolutional neural networks).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from . import config


def set_global_seed(seed: int = config.RANDOM_SEED) -> None:
    """Seed Python, NumPy and TensorFlow for reproducible experiments."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _convolution_block(
    tensor: tf.Tensor,
    filters: int,
    block_index: int,
    dropout_rate: float = 0.25,
) -> tf.Tensor:
    """Append a convolution, normalisation, pooling and dropout block.

    Batch normalisation is applied before the activation so that the input to
    the non linearity has a stable distribution across mini batches, which
    permits a higher learning rate. Dropout after pooling provides stochastic
    regularisation that reduces co-adaptation between feature maps.
    """
    tensor = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"block{block_index}_conv",
    )(tensor)
    tensor = layers.BatchNormalization(name=f"block{block_index}_bn")(tensor)
    tensor = layers.Activation("relu", name=f"block{block_index}_relu")(tensor)
    tensor = layers.MaxPooling2D(pool_size=(2, 2), name=f"block{block_index}_pool")(tensor)
    tensor = layers.Dropout(dropout_rate, name=f"block{block_index}_dropout")(tensor)
    return tensor


def build_cnn(
    input_shape: Tuple[int, int, int] = config.IMAGE_SHAPE,
    num_classes: int = config.NUM_CLASSES,
    filters: Tuple[int, ...] = (32, 64, 128, 256),
    dense_units: int = 512,
    name: str = "brain_tumor_cnn",
) -> models.Model:
    """Construct the custom convolutional classifier.

    The network uses four convolution blocks with doubling filter counts,
    followed by global average pooling and a single fully connected layer.
    Global average pooling is preferred over flattening because it reduces the
    parameter count by two orders of magnitude and acts as a structural
    regulariser.

    Returns
    -------
    tensorflow.keras.Model
        An uncompiled model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    tensor = inputs
    for index, filter_count in enumerate(filters, start=1):
        tensor = _convolution_block(tensor, filter_count, index)

    tensor = layers.GlobalAveragePooling2D(name="global_average_pooling")(tensor)
    tensor = layers.Dense(
        dense_units,
        use_bias=False,
        kernel_initializer="he_normal",
        name="dense_features",
    )(tensor)
    tensor = layers.BatchNormalization(name="dense_bn")(tensor)
    tensor = layers.Activation("relu", name="dense_relu")(tensor)
    tensor = layers.Dropout(0.5, name="dense_dropout")(tensor)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(tensor)

    return models.Model(inputs=inputs, outputs=outputs, name=name)


def build_transfer_model(
    input_shape: Tuple[int, int, int] = config.IMAGE_SHAPE,
    num_classes: int = config.NUM_CLASSES,
    trainable_base: bool = False,
    name: str = "mobilenetv2_transfer",
) -> models.Model:
    """Construct a transfer learning classifier on a MobileNetV2 backbone.

    This model is included as an additional comparison point. It demonstrates
    how representations learned on a large natural image corpus transfer to
    medical imaging, and it serves as a contingency should the custom network
    underperform.
    """
    from tensorflow.keras.applications import MobileNetV2

    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = trainable_base

    inputs = layers.Input(shape=input_shape, name="input_image")
    # MobileNetV2 expects inputs scaled to the interval minus one to one.
    tensor = layers.Rescaling(scale=2.0, offset=-1.0, name="rescale")(inputs)
    tensor = base(tensor, training=False)
    tensor = layers.GlobalAveragePooling2D(name="global_average_pooling")(tensor)
    tensor = layers.Dropout(0.3, name="head_dropout")(tensor)
    tensor = layers.Dense(256, activation="relu", name="head_dense")(tensor)
    tensor = layers.Dropout(0.3, name="head_dropout_2")(tensor)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(tensor)

    return models.Model(inputs=inputs, outputs=outputs, name=name)


def compile_model(
    model: models.Model,
    learning_rate: float = config.LEARNING_RATE,
    label_smoothing: float = 0.0,
) -> models.Model:
    """Compile a model with the Adam optimiser and categorical cross entropy.

    Adam combines momentum with per parameter learning rate scaling, which
    converges faster than plain stochastic gradient descent on this dataset.
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=False),
        ],
    )
    return model


def build_callbacks(
    checkpoint_path: Path | str,
    monitor: str = "val_accuracy",
    early_stopping_patience: int = config.EARLY_STOPPING_PATIENCE,
) -> List[Callback]:
    """Assemble the standard training callbacks.

    Early stopping prevents the network from continuing past the point of
    optimal generalisation, the learning rate scheduler escapes plateaus by
    halving the step size, and the checkpoint retains the best weights
    observed on the validation split.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "max" if "accuracy" in monitor or "auc" in monitor else "min"

    return [
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LEARNING_RATE,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            verbose=1,
        ),
    ]


def summarise_architecture(model: models.Model):
    """Return a table describing every layer of a model."""
    import pandas as pd

    records = []
    for layer in model.layers:
        try:
            output_shape = layer.output.shape
        except AttributeError:
            output_shape = None
        records.append(
            {
                "layer": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": str(output_shape),
                "parameters": layer.count_params(),
            }
        )
    return pd.DataFrame.from_records(records)
