"""
Streamlit interface for the brain tumour classification system.

The application exposes four views: an overview of the study, an inference
page that accepts an uploaded magnetic resonance slice, a results dashboard
that renders the metrics produced by the training notebooks, and a methodology
page that documents the pipeline.

Run locally with::

    streamlit run app/streamlit_app.py

Course: DS 4889 Machine Learning, United International University.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

st.set_page_config(
    page_title="Brain Tumour MRI Classification",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the trained model.")
def load_model() -> Tuple[Optional[object], Optional[str]]:
    """Load the first available trained model from the models directory."""
    candidates = [
        config.BEST_MODEL_NAME,
        config.HYBRID_MODEL_NAME,
        config.BASELINE_MODEL_NAME,
        config.TRANSFER_MODEL_NAME,
    ]
    try:
        import tensorflow as tf
    except ImportError:
        return None, None

    for filename in candidates:
        path = config.MODELS_DIR / filename
        if path.exists():
            try:
                return tf.keras.models.load_model(path), filename
            except Exception:
                continue
    return None, None


@st.cache_data(show_spinner=False)
def load_metrics_file(filename: str) -> Optional[Dict]:
    """Read a JSON metrics artefact if the training stage produced it."""
    path = config.METRICS_DIR / filename
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(show_spinner=False)
def load_comparison_table() -> Optional[pd.DataFrame]:
    """Read the experiment comparison table if it exists."""
    for name in ("final_comparison.csv", "experiment_comparison.csv"):
        path = config.METRICS_DIR / name
        if path.exists():
            try:
                return pd.read_csv(path)
            except (pd.errors.ParserError, OSError):
                continue
    return None


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def run_inference(model, image_array: np.ndarray, use_segmentation: bool) -> Dict[str, object]:
    """Preprocess an image, optionally segment it, then classify it."""
    from src import gmm_segmentation, preprocessing

    processed = preprocessing.preprocess_image(image_array, config.IMAGE_SIZE)

    segmentation = None
    model_input = processed
    if use_segmentation:
        segmentation = gmm_segmentation.segment_image(processed)
        model_input = gmm_segmentation.apply_attention_mask(processed, segmentation["mask"])

    start = time.perf_counter()
    probabilities = model.predict(np.expand_dims(model_input, axis=0), verbose=0)[0]
    elapsed = time.perf_counter() - start

    predicted_index = int(np.argmax(probabilities))

    return {
        "processed": processed,
        "model_input": model_input,
        "segmentation": segmentation,
        "probabilities": probabilities,
        "predicted_index": predicted_index,
        "predicted_label": config.CLASS_LABELS[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "latency_ms": elapsed * 1000.0,
    }


def render_probability_chart(probabilities: np.ndarray) -> go.Figure:
    """Build a horizontal bar chart of class probabilities."""
    frame = pd.DataFrame(
        {
            "Class": config.CLASS_LABELS,
            "Probability": [float(value) for value in probabilities],
        }
    ).sort_values("Probability")

    figure = px.bar(
        frame,
        x="Probability",
        y="Class",
        orientation="h",
        text=[f"{value:.1%}" for value in frame["Probability"]],
        color="Class",
        color_discrete_sequence=config.CLASS_COLORS,
    )
    figure.update_layout(
        showlegend=False,
        xaxis_range=[0, 1],
        xaxis_tickformat=".0%",
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    figure.update_traces(textposition="outside", cliponaxis=False)
    return figure


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def view_overview() -> None:
    """Project summary, dataset description and research question."""
    st.title("Brain Tumour Classification from Magnetic Resonance Imaging")
    st.caption(
        "A hybrid pipeline combining Gaussian Mixture Model segmentation with "
        "convolutional classification."
    )

    st.markdown(
        """
        This application accompanies a study carried out for DS 4889 Machine
        Learning at United International University. The study investigates
        whether an unsupervised intensity model, fitted independently to each
        magnetic resonance slice, can supply a useful spatial prior to a
        convolutional classifier.
        """
    )

    st.subheader("Research question")
    st.info(
        "Does unsupervised Gaussian Mixture Model pre-segmentation improve the "
        "accuracy of convolutional brain tumour classification, relative to an "
        "otherwise identical network trained on unsegmented input?"
    )

    st.subheader("Dataset")
    dataset_frame = pd.DataFrame(
        {
            "Class": config.CLASS_LABELS,
            "Training images": [1400, 1400, 1400, 1400],
            "Testing images": [400, 400, 400, 400],
        }
    )
    dataset_frame["Total"] = (
        dataset_frame["Training images"] + dataset_frame["Testing images"]
    )

    left, right = st.columns([3, 2])
    with left:
        st.dataframe(dataset_frame, hide_index=True, use_container_width=True)
        st.caption(
            "Source: Brain Tumor MRI Dataset, Masoud Nickparvar, Kaggle. "
            "Total 7,200 images, balanced across the four classes."
        )
    with right:
        figure = px.pie(
            dataset_frame,
            names="Class",
            values="Total",
            color="Class",
            color_discrete_sequence=config.CLASS_COLORS,
            hole=0.45,
        )
        figure.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(figure, use_container_width=True)

    st.subheader("Diagnostic categories")
    for label in config.CLASS_LABELS:
        with st.expander(label):
            st.write(config.CLASS_DESCRIPTIONS[label])

    st.subheader("Processing pipeline")
    st.markdown(
        """
        1. **Acquisition.** A single axial slice is supplied as a standard image file.
        2. **Preprocessing.** The slice is resized to 224 by 224 pixels and enhanced
           with Contrast Limited Adaptive Histogram Equalisation, which raises local
           contrast at lesion boundaries without amplifying global noise.
        3. **Unsupervised segmentation.** A three component Gaussian Mixture Model is
           fitted to the intensity histogram by Expectation Maximisation. The component
           corresponding to hyperintense tissue is selected under an area constraint and
           converted into a soft attention map.
        4. **Classification.** A four block convolutional network with batch
           normalisation and dropout produces a probability distribution over the four
           diagnostic categories.
        5. **Explanation.** Gradient weighted Class Activation Mapping localises the
           evidence supporting the prediction.
        """
    )

    st.warning(config.MEDICAL_DISCLAIMER)


def view_inference() -> None:
    """Upload an image and obtain a prediction with a visual explanation."""
    st.title("Image Analysis")
    st.caption("Upload a brain magnetic resonance slice to obtain a classification.")

    model, model_name = load_model()

    if model is None:
        st.error(
            "No trained model was found in the models directory. Run the training "
            "notebooks, or execute `python -m src.train`, to produce a model "
            "artefact before using this page."
        )
        st.stop()

    st.success(f"Active model artefact: {model_name}")

    control_left, control_right = st.columns(2)
    with control_left:
        use_segmentation = st.toggle(
            "Apply Gaussian Mixture Model attention",
            value=model_name == config.HYBRID_MODEL_NAME,
            help=(
                "Enable this when the loaded model was trained on segmented input. "
                "Disabling it feeds the preprocessed image directly to the network."
            ),
        )
    with control_right:
        show_explanation = st.toggle("Generate Grad-CAM explanation", value=True)

    uploaded = st.file_uploader(
        "Magnetic resonance slice",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        help="A single axial slice in any common raster format.",
    )

    if uploaded is None:
        st.info(
            "Select a file to begin. Images from the Testing split of the dataset are "
            "suitable for verifying the system."
        )
        return

    image = Image.open(uploaded).convert("RGB")
    image_array = np.array(image)

    with st.spinner("Running inference."):
        outcome = run_inference(model, image_array, use_segmentation)

    st.divider()

    summary_columns = st.columns(3)
    summary_columns[0].metric("Predicted class", outcome["predicted_label"])
    summary_columns[1].metric("Confidence", f"{outcome['confidence']:.1%}")
    summary_columns[2].metric("Latency", f"{outcome['latency_ms']:.0f} ms")

    if outcome["predicted_label"] == "No Tumor":
        st.success("No tumour was detected in the supplied slice.")
    else:
        st.warning(
            f"A finding consistent with {outcome['predicted_label'].lower()} was "
            "detected. Radiological confirmation is required."
        )

    if outcome["confidence"] < 0.60:
        st.info(
            "The confidence of this prediction is low. Low confidence commonly "
            "indicates an image that falls outside the distribution the model was "
            "trained on, and the output should not be relied upon."
        )

    st.plotly_chart(render_probability_chart(outcome["probabilities"]), use_container_width=True)

    st.subheader("Visual record")
    image_columns = st.columns(4 if show_explanation else 3)

    image_columns[0].image(image, caption="Uploaded image", use_container_width=True)
    image_columns[1].image(
        outcome["processed"], caption="Preprocessed", use_container_width=True
    )

    if outcome["segmentation"] is not None:
        image_columns[2].image(
            outcome["segmentation"]["mask"],
            caption="Segmentation mask",
            use_container_width=True,
        )
    else:
        image_columns[2].image(
            outcome["model_input"], caption="Network input", use_container_width=True
        )

    if show_explanation:
        try:
            from src import explainability

            explanation = explainability.generate_explanation(
                model, outcome["model_input"], outcome["predicted_index"]
            )
            image_columns[3].image(
                explanation["overlay"],
                caption="Grad-CAM attribution",
                use_container_width=True,
            )
        except Exception as error:
            image_columns[3].warning(f"Explanation unavailable: {error}")

    st.subheader("Interpretation")
    st.write(config.CLASS_DESCRIPTIONS[outcome["predicted_label"]])

    with st.expander("Full probability table"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Class": config.CLASS_LABELS,
                    "Probability": [f"{value:.4f}" for value in outcome["probabilities"]],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.warning(config.MEDICAL_DISCLAIMER)


def view_results() -> None:
    """Render the metrics generated by the training and evaluation notebooks."""
    st.title("Experimental Results")
    st.caption("Figures are read from the artefacts written by the evaluation notebooks.")

    comparison = load_comparison_table()

    if comparison is None:
        st.warning(
            "No results have been generated yet. Run notebooks 03 to 05, or execute "
            "`python -m src.train`, to populate the results directory."
        )
        st.subheader("Expected reporting format")
        st.dataframe(
            pd.DataFrame(
                {
                    "model": ["CNN baseline", "GMM and CNN hybrid"],
                    "accuracy": ["pending", "pending"],
                    "precision_weighted": ["pending", "pending"],
                    "recall_weighted": ["pending", "pending"],
                    "f1_weighted": ["pending", "pending"],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        return

    st.subheader("Model comparison")
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    metric_columns = [
        column
        for column in ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted")
        if column in comparison.columns
    ]

    if metric_columns and "model" in comparison.columns:
        melted = comparison.melt(
            id_vars="model",
            value_vars=metric_columns,
            var_name="Metric",
            value_name="Score",
        )
        figure = px.bar(
            melted,
            x="Metric",
            y="Score",
            color="model",
            barmode="group",
            text=[f"{value:.3f}" for value in melted["Score"]],
        )
        figure.update_layout(yaxis_range=[0, 1.05], height=420)
        figure.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(figure, use_container_width=True)

    significance = load_metrics_file("mcnemar_test.json")
    if significance is not None:
        st.subheader("Paired significance test")
        st.caption(
            "McNemar's test compares two classifiers evaluated on identical samples "
            "by examining only the cases where exactly one of them is correct."
        )
        test_columns = st.columns(4)
        test_columns[0].metric("Both correct", significance.get("both_correct", 0))
        test_columns[1].metric(
            "Only model A correct", significance.get("only_model_a_correct", 0)
        )
        test_columns[2].metric(
            "Only model B correct", significance.get("only_model_b_correct", 0)
        )
        test_columns[3].metric("p value", f"{significance.get('p_value', float('nan')):.4f}")

        if significance.get("significant_at_0.05"):
            st.success(
                "The difference between the two models is statistically significant "
                "at the five percent level."
            )
        else:
            st.info(
                "The difference between the two models is not statistically "
                "significant at the five percent level."
            )

    figures_directory = config.FIGURES_DIR
    available_figures = sorted(figures_directory.glob("*.png")) if figures_directory.exists() else []

    if available_figures:
        st.subheader("Generated figures")
        selected = st.selectbox(
            "Figure",
            available_figures,
            format_func=lambda path: path.stem.replace("_", " ").title(),
        )
        st.image(str(selected), use_container_width=True)


def view_methodology() -> None:
    """Document the method and map it onto the course syllabus."""
    st.title("Methodology")

    st.subheader("Gaussian Mixture Model")
    st.markdown(
        "The intensity histogram of a slice is modelled as a weighted sum of "
        "Gaussian densities:"
    )
    st.latex(r"p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)")
    st.markdown(
        """
        The parameters are estimated by Expectation Maximisation. In the
        expectation step the responsibility of component k for observation x is

        """
    )
    st.latex(
        r"\gamma_k(x) = \frac{\pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)}"
        r"{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x \mid \mu_j, \Sigma_j)}"
    )
    st.markdown(
        """
        In the maximisation step the mixing weights, means and covariances are
        updated to their weighted maximum likelihood estimates. Because the
        principal tissue classes of a magnetic resonance slice occupy broadly
        separable intensity bands, the fitted components approximate background,
        normal parenchyma and hyperintense tissue.

        The component corresponding to a lesion candidate is selected by a scoring
        rule that rewards high mean intensity but penalises components occupying an
        implausibly large or small fraction of the slice, which prevents the skull
        or the image background from being selected.
        """
    )

    st.subheader("Soft attention rather than hard masking")
    st.markdown(
        """
        Removing everything outside the mask would discard the anatomical context
        required to distinguish a meningioma at the brain margin from a pituitary
        lesion at the skull base. The binary mask is therefore blurred into a
        continuous attention map and the background is retained at reduced
        intensity. The network is biased towards the region of interest while
        global structure remains available.
        """
    )

    st.subheader("Network architecture")
    architecture = pd.DataFrame(
        {
            "Stage": [
                "Block 1",
                "Block 2",
                "Block 3",
                "Block 4",
                "Pooling",
                "Dense",
                "Output",
            ],
            "Composition": [
                "Conv2D 32 filters, batch normalisation, ReLU, max pooling, dropout 0.25",
                "Conv2D 64 filters, batch normalisation, ReLU, max pooling, dropout 0.25",
                "Conv2D 128 filters, batch normalisation, ReLU, max pooling, dropout 0.25",
                "Conv2D 256 filters, batch normalisation, ReLU, max pooling, dropout 0.25",
                "Global average pooling",
                "512 units, batch normalisation, ReLU, dropout 0.50",
                "4 units, softmax",
            ],
            "Rationale": [
                "Edge and texture primitives",
                "Local shape composition",
                "Regional structure",
                "Lesion level abstraction",
                "Parameter reduction and structural regularisation",
                "Decision boundary in feature space",
                "Probability distribution over classes",
            ],
        }
    )
    st.dataframe(architecture, hide_index=True, use_container_width=True)

    st.subheader("Training protocol")
    protocol = pd.DataFrame(
        {
            "Setting": [
                "Optimiser",
                "Initial learning rate",
                "Learning rate schedule",
                "Batch size",
                "Maximum epochs",
                "Early stopping",
                "Loss function",
                "Class weighting",
                "Validation split",
                "Augmentation",
            ],
            "Value": [
                "Adam",
                f"{config.LEARNING_RATE}",
                f"Halved after {config.REDUCE_LR_PATIENCE} epochs without improvement",
                f"{config.BATCH_SIZE}",
                f"{config.EPOCHS}",
                f"Patience {config.EARLY_STOPPING_PATIENCE}, best weights restored",
                "Categorical cross entropy",
                "Inverse class frequency, mean normalised",
                f"{int(config.VALIDATION_SPLIT * 100)} percent, stratified",
                "Rotation, translation, zoom, horizontal flip",
            ],
        }
    )
    st.dataframe(protocol, hide_index=True, use_container_width=True)

    st.subheader("Course syllabus coverage")
    syllabus = pd.DataFrame(
        {
            "Syllabus topic": [
                "Supervised learning",
                "Unsupervised learning",
                "Classification techniques",
                "Maximum likelihood estimation",
                "Gaussian Mixture Model and Expectation Maximisation",
                "Principal Component Analysis",
                "Ensemble methods",
                "Neural networks and backpropagation",
                "Batch normalisation, dropout, optimisers",
                "Convolutional neural networks",
                "Evaluation and cross validation",
            ],
            "Application in this study": [
                "Convolutional classifier trained on labelled slices",
                "Mixture model fitted per image without labels",
                "Logistic regression, naive Bayes and softmax output layer",
                "Mixture parameters estimated by maximum likelihood",
                "Core of the segmentation stage",
                "Dimensionality reduction for the classical baseline and for visualisation",
                "Random forest and stacking baselines on engineered descriptors",
                "Network training and Grad-CAM attribution",
                "Applied throughout the architecture and training protocol",
                "Primary classification model",
                "Stratified splits, bootstrap intervals, McNemar's paired test",
            ],
        }
    )
    st.dataframe(syllabus, hide_index=True, use_container_width=True)

    st.subheader("Limitations")
    st.markdown(
        """
        - A single publicly available dataset was used, so generalisation across
          scanners, acquisition protocols and populations is unverified.
        - Analysis is restricted to two dimensional slices rather than full volumes.
        - The four category taxonomy is coarse relative to clinical tumour grading.
        - No radiologist adjudication or prospective evaluation was performed.
        - Predicted probabilities have not been calibrated, so confidence values
          should not be read as true posterior probabilities.
        """
    )

    st.warning(config.MEDICAL_DISCLAIMER)


# ---------------------------------------------------------------------------
# Application shell
# ---------------------------------------------------------------------------


VIEWS = {
    "Overview": view_overview,
    "Image Analysis": view_inference,
    "Results": view_results,
    "Methodology": view_methodology,
}


def main() -> None:
    """Render the sidebar and dispatch to the selected view."""
    st.sidebar.title("Brain Tumour Classification")
    st.sidebar.caption("DS 4889 Machine Learning")

    selection = st.sidebar.radio("Section", list(VIEWS), label_visibility="collapsed")

    st.sidebar.divider()

    _, model_name = load_model()
    if model_name:
        st.sidebar.success(f"Model loaded\n\n{model_name}")
    else:
        st.sidebar.error("No trained model found")

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        **Institution**
        United International University

        **Department**
        Computer Science and Engineering

        **Dataset**
        [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
        """
    )
    st.sidebar.divider()
    st.sidebar.caption(config.MEDICAL_DISCLAIMER)

    VIEWS[selection]()


if __name__ == "__main__":
    main()
