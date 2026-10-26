"""
Command line training entry point.

Running this module reproduces the complete experimental protocol described in
the project report: the baseline convolutional network, the hybrid pipeline in
which a Gaussian Mixture Model supplies a spatial prior, an optional transfer
learning comparison and an optional classical feature based baseline.

Example
-------
Train both convolutional experiments for forty epochs::

    python -m src.train --experiments baseline hybrid --epochs 40

Run a fast smoke test on two hundred images per class::

    python -m src.train --experiments baseline --epochs 3 --limit-per-class 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from tensorflow.keras.utils import to_categorical

from . import classical_models, config, data_loader, evaluate, gmm_segmentation, preprocessing
from .cnn_model import build_callbacks, build_cnn, build_transfer_model, compile_model, set_global_seed


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_arrays(
    limit_per_class: int | None,
    image_size: int,
    data_dir: Path | str | None = None,
) -> Dict[str, np.ndarray]:
    """Load the dataset, apply preprocessing and carve out a validation split."""
    print("Loading raw images from disk.")
    raw = data_loader.load_dataset(
        data_dir=data_dir, image_size=image_size, limit_per_class=limit_per_class
    )

    print("Applying preprocessing chain.")
    x_train_full = preprocessing.preprocess_batch(raw["x_train"], image_size)
    x_test = preprocessing.preprocess_batch(raw["x_test"], image_size)

    x_train, x_validation, y_train, y_validation = data_loader.stratified_validation_split(
        x_train_full, raw["y_train"]
    )

    print(
        f"Training samples {len(x_train)}, validation samples {len(x_validation)}, "
        f"test samples {len(x_test)}."
    )

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_validation": x_validation,
        "y_validation": y_validation,
        "x_test": x_test,
        "y_test": raw["y_test"],
    }


def apply_gmm_stage(arrays: Dict[str, np.ndarray], use_cache: bool) -> Dict[str, np.ndarray]:
    """Transform every split with the Gaussian Mixture Model attention stage."""
    cache_file = config.CACHE_DIR / "hybrid_arrays.npz"

    if use_cache and cache_file.exists():
        print(f"Loading cached segmentation output from {cache_file}.")
        cached = np.load(cache_file)
        if cached["x_train"].shape == arrays["x_train"].shape:
            return {
                "x_train": cached["x_train"],
                "x_validation": cached["x_validation"],
                "x_test": cached["x_test"],
            }
        print("Cached arrays do not match the current configuration. Recomputing.")

    transformed = {
        "x_train": gmm_segmentation.transform_batch_for_hybrid(
            arrays["x_train"], description="Segmenting training split"
        ),
        "x_validation": gmm_segmentation.transform_batch_for_hybrid(
            arrays["x_validation"], description="Segmenting validation split"
        ),
        "x_test": gmm_segmentation.transform_batch_for_hybrid(
            arrays["x_test"], description="Segmenting test split"
        ),
    }

    if use_cache:
        np.savez_compressed(cache_file, **transformed)
        print(f"Cached segmentation output to {cache_file}.")

    return transformed


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def run_convolutional_experiment(
    experiment_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    model_filename: str,
    epochs: int,
    batch_size: int,
    use_transfer_learning: bool = False,
    use_class_weights: bool = True,
) -> Dict[str, object]:
    """Train one convolutional model and evaluate it on the held out test set."""
    print(f"\nStarting experiment: {experiment_name}")
    set_global_seed()

    y_train_categorical = to_categorical(y_train, config.NUM_CLASSES)
    y_validation_categorical = to_categorical(y_validation, config.NUM_CLASSES)

    model = build_transfer_model() if use_transfer_learning else build_cnn()
    model = compile_model(model)
    print(f"Trainable parameters: {model.count_params():,}")

    checkpoint_path = config.MODELS_DIR / model_filename
    callbacks = build_callbacks(checkpoint_path)

    generator = preprocessing.make_training_generator(
        x_train, y_train_categorical, batch_size=batch_size
    )
    steps_per_epoch = max(1, len(x_train) // batch_size)

    class_weight = data_loader.compute_class_weights(y_train) if use_class_weights else None

    start_time = time.time()
    history = model.fit(
        generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=(x_validation, y_validation_categorical),
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    training_duration = time.time() - start_time

    print(f"Training finished in {training_duration / 60:.1f} minutes.")

    inference_start = time.time()
    probabilities = model.predict(x_test, batch_size=batch_size, verbose=0)
    inference_duration = time.time() - inference_start

    predictions = np.argmax(probabilities, axis=1)

    metrics = evaluate.compute_metrics(y_test, predictions, experiment_name)
    metrics["training_time_seconds"] = round(training_duration, 2)
    metrics["inference_time_per_image_ms"] = round(1000 * inference_duration / len(x_test), 3)
    metrics["epochs_completed"] = len(history.history["loss"])
    metrics["trainable_parameters"] = int(model.count_params())

    print(
        f"Test accuracy {metrics['accuracy']:.4f}, "
        f"weighted F1 {metrics['f1_weighted']:.4f}."
    )

    return {
        "name": experiment_name,
        "model": model,
        "history": history.history,
        "probabilities": probabilities,
        "predictions": predictions,
        "metrics": metrics,
    }


def run_classical_experiment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, object]:
    """Train the engineered feature baselines on Gaussian Mixture descriptors."""
    print("\nStarting experiment: classical feature baselines")

    train_features = gmm_segmentation.extract_gmm_features_batch(x_train)
    test_features = gmm_segmentation.extract_gmm_features_batch(x_test)

    pca, train_scores = gmm_segmentation.fit_pca(x_train)
    test_scores = gmm_segmentation.transform_pca(pca, x_test)

    train_matrix = np.hstack([train_features, train_scores])
    test_matrix = np.hstack([test_features, test_scores])

    print(f"Feature matrix shape: {train_matrix.shape}")

    results = classical_models.train_classical_models(
        train_matrix, y_train, test_matrix, y_test
    )

    metric_rows = []
    for name, result in results.items():
        row = evaluate.compute_metrics(y_test, result.predictions, f"classical_{name}")
        metric_rows.append(row)
        print(f"  {name}: accuracy {row['accuracy']:.4f}, weighted F1 {row['f1_weighted']:.4f}")

    best_name = max(results, key=lambda key: results[key].f1_weighted)
    classical_models.save_estimator(
        results[best_name].estimator, config.MODELS_DIR / config.CLASSICAL_MODEL_NAME
    )

    return {"results": results, "metrics": metric_rows, "best": best_name}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Define and read the command line interface."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate brain tumour classification models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["baseline", "hybrid"],
        choices=["baseline", "hybrid", "transfer", "classical"],
        help="Which experiments to run.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root folder containing the Training and Testing splits.",
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Cap the number of images loaded per class for quick runs.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Recompute the segmentation stage instead of reusing cached arrays.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse frequency class weighting.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the selected experiments and write a comparison report."""
    arguments = parse_arguments()
    set_global_seed()

    print("Verifying dataset layout.")
    inventory = data_loader.verify_dataset(arguments.data_dir)
    print(inventory.to_string(index=False))

    arrays = prepare_arrays(
        arguments.limit_per_class, arguments.image_size, arguments.data_dir
    )

    completed: List[Dict[str, object]] = []
    all_metrics: List[Dict[str, float]] = []

    if "baseline" in arguments.experiments:
        result = run_convolutional_experiment(
            experiment_name="CNN baseline",
            x_train=arrays["x_train"],
            y_train=arrays["y_train"],
            x_validation=arrays["x_validation"],
            y_validation=arrays["y_validation"],
            x_test=arrays["x_test"],
            y_test=arrays["y_test"],
            model_filename=config.BASELINE_MODEL_NAME,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            use_class_weights=not arguments.no_class_weights,
        )
        completed.append(result)
        all_metrics.append(result["metrics"])

    if "hybrid" in arguments.experiments:
        segmented = apply_gmm_stage(arrays, use_cache=not arguments.no_cache)
        result = run_convolutional_experiment(
            experiment_name="GMM and CNN hybrid",
            x_train=segmented["x_train"],
            y_train=arrays["y_train"],
            x_validation=segmented["x_validation"],
            y_validation=arrays["y_validation"],
            x_test=segmented["x_test"],
            y_test=arrays["y_test"],
            model_filename=config.HYBRID_MODEL_NAME,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            use_class_weights=not arguments.no_class_weights,
        )
        completed.append(result)
        all_metrics.append(result["metrics"])

    if "transfer" in arguments.experiments:
        result = run_convolutional_experiment(
            experiment_name="MobileNetV2 transfer",
            x_train=arrays["x_train"],
            y_train=arrays["y_train"],
            x_validation=arrays["x_validation"],
            y_validation=arrays["y_validation"],
            x_test=arrays["x_test"],
            y_test=arrays["y_test"],
            model_filename=config.TRANSFER_MODEL_NAME,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            use_transfer_learning=True,
            use_class_weights=not arguments.no_class_weights,
        )
        completed.append(result)
        all_metrics.append(result["metrics"])

    if "classical" in arguments.experiments:
        classical_outcome = run_classical_experiment(
            arrays["x_train"], arrays["y_train"], arrays["x_test"], arrays["y_test"]
        )
        all_metrics.extend(classical_outcome["metrics"])

    if not all_metrics:
        print("No experiments were selected.")
        return

    comparison = evaluate.build_comparison_table(all_metrics)
    print("\nFinal comparison")
    print(comparison.to_string(index=False))

    evaluate.save_metrics(all_metrics, "experiment_metrics.json")
    comparison.to_csv(config.METRICS_DIR / "experiment_comparison.csv", index=False)

    # A paired significance test is only meaningful between two neural models
    # evaluated on identical samples.
    if len(completed) >= 2:
        test_result = evaluate.mcnemar_test(
            arrays["y_test"],
            completed[0]["predictions"],
            completed[1]["predictions"],
        )
        print("\nMcNemar paired comparison")
        print(json.dumps(test_result, indent=2))
        evaluate.save_metrics(test_result, "mcnemar_test.json")

    best_result = max(completed, key=lambda item: item["metrics"]["f1_weighted"]) if completed else None
    if best_result is not None:
        best_path = config.MODELS_DIR / config.BEST_MODEL_NAME
        best_result["model"].save(best_path)
        print(f"\nBest model '{best_result['name']}' saved to {best_path}.")


if __name__ == "__main__":
    main()
