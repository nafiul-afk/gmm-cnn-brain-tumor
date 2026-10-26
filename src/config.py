"""
Central configuration for the Brain Tumor Detection project.

All tunable constants are defined here so that notebooks, training scripts and
the Streamlit application share a single source of truth. Changing a value in
this module propagates to every stage of the pipeline.

Course: DS 4889 Machine Learning, United International University.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
CACHE_DIR = PROJECT_ROOT / "cache"

for _directory in (MODELS_DIR, FIGURES_DIR, METRICS_DIR, CACHE_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

# Folder names exactly as they appear in the Kaggle dataset.
CLASS_FOLDERS = ["glioma", "meningioma", "notumor", "pituitary"]

# Human readable labels used in reports, plots and the web interface.
CLASS_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

NUM_CLASSES = len(CLASS_FOLDERS)

LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASS_FOLDERS)}
INDEX_TO_LABEL = {index: name for name, index in LABEL_TO_INDEX.items()}

CLASS_DESCRIPTIONS = {
    "Glioma": (
        "A tumour originating in the glial cells that support neurons in the "
        "brain and spinal cord. Gliomas vary in grade and may be benign or "
        "malignant."
    ),
    "Meningioma": (
        "A tumour arising from the meninges, the protective membranes covering "
        "the brain and spinal cord. The majority of meningiomas are benign."
    ),
    "No Tumor": (
        "No abnormal mass detected. The scan presents anatomy consistent with "
        "a healthy brain."
    ),
    "Pituitary": (
        "A tumour forming in the pituitary gland at the base of the brain. "
        "Most pituitary tumours are benign adenomas."
    ),
}

# Colour palette reused across every figure for visual consistency.
CLASS_COLORS = ["#C0392B", "#D68910", "#1E8449", "#1F4E79"]

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

IMAGE_SIZE = 224
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Contrast Limited Adaptive Histogram Equalisation parameters.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# ---------------------------------------------------------------------------
# Gaussian Mixture Model segmentation
# ---------------------------------------------------------------------------

# Three components approximate background, healthy brain tissue and the
# hyperintense region that commonly corresponds to a lesion.
GMM_N_COMPONENTS = 3
GMM_COVARIANCE_TYPE = "full"
GMM_MAX_ITER = 100

# Morphological cleanup applied to the binary lesion mask.
MORPH_KERNEL_SIZE = 5

# Fraction of the image the mask may occupy before it is rejected as a
# background artefact rather than a lesion candidate.
MASK_MAX_AREA_RATIO = 0.45
MASK_MIN_AREA_RATIO = 0.005

# Weight of the original image retained when blending the mask back in. A value
# of 1.0 would discard all context outside the region of interest.
ROI_BACKGROUND_WEIGHT = 0.25

# ---------------------------------------------------------------------------
# Principal Component Analysis
# ---------------------------------------------------------------------------

PCA_N_COMPONENTS = 50
PCA_IMAGE_SIZE = 64  # Images are downscaled before flattening for PCA.

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

# Data augmentation ranges applied to the training split only.
AUGMENTATION = {
    "rotation_range": 20,
    "width_shift_range": 0.10,
    "height_shift_range": 0.10,
    "zoom_range": 0.20,
    "horizontal_flip": True,
    "fill_mode": "nearest",
}

# ---------------------------------------------------------------------------
# Model artefact names
# ---------------------------------------------------------------------------

BASELINE_MODEL_NAME = "cnn_baseline.keras"
HYBRID_MODEL_NAME = "gmm_cnn_hybrid.keras"
TRANSFER_MODEL_NAME = "transfer_mobilenetv2.keras"
BEST_MODEL_NAME = "best_model.keras"
CLASSICAL_MODEL_NAME = "gmm_pca_classical.joblib"

MEDICAL_DISCLAIMER = (
    "This system was developed for academic coursework and research "
    "demonstration only. It is not a certified medical device and must not be "
    "used for clinical diagnosis. All findings require confirmation by a "
    "qualified radiologist."
)
