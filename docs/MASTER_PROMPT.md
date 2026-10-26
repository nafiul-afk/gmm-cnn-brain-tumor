# Master Context Brief

Paste the section below into any AI assistant to give it complete context about
this project. Append your specific request at the end under "Request".

---

## BEGIN CONTEXT

You are assisting with an academic machine learning project submitted for
DS 4889 Machine Learning at United International University, Department of
Computer Science and Engineering. Respond in formal technical English. Do not
use emojis. Do not use em-dashes or en-dashes; use commas, colons or the word
"to" for ranges.

### Project

Brain tumour classification from magnetic resonance imaging using a hybrid
pipeline that combines unsupervised Gaussian Mixture Model segmentation with
supervised convolutional classification.

### Research question

Does unsupervised Gaussian Mixture Model pre-segmentation improve the accuracy
of convolutional brain tumour classification, relative to an otherwise identical
network trained on unsegmented input?

The hypothesis is that supplying a spatial prior reduces the burden on the
network of learning where to attend from labels alone. The study is designed as
a controlled comparison: architecture, seeds, partitions, augmentation and
training schedule are held constant, and only the input transformation varies.
A negative result is acceptable and must be reported honestly.

### Dataset

- Brain Tumor MRI Dataset, Masoud Nickparvar, Kaggle
- Identifier `masoudnickparvar/brain-tumor-mri-dataset`
- 7,200 JPEG images
- Four classes with folder names `glioma`, `meningioma`, `notumor`, `pituitary`
- Display labels `Glioma`, `Meningioma`, `No Tumor`, `Pituitary`
- Layout `data/Training/<class>/` and `data/Testing/<class>/`
- Training 5,600 images, Testing 1,600 images
- Per class training counts 1400, 1400, 1400, 1400
- Per class testing counts 400, 400, 400, 400
- Exactly balanced, imbalance ratio 1.00 to 1

Note: many published descriptions quote 7,023 images with unequal class counts.
Those figures refer to an earlier revision of the dataset. The counts above were
enumerated directly from the archive on 26 October 2026. Because the dataset is
balanced, class weighting is inert and weighted metrics coincide with macro
metrics; both are still reported so the analysis survives a future revision.

### Partitions

- Training: 80 percent of the Training folder, stratified
- Validation: 20 percent of the Training folder, stratified, used for early
  stopping and learning rate scheduling
- Test: the Testing folder, evaluated exactly once per model

### Preprocessing

1. Resize to 224 by 224 pixels using area interpolation
2. Contrast Limited Adaptive Histogram Equalisation, clip limit 2.0, tile grid
   8 by 8, applied on the luminance channel
3. Divide by 255 to place values in the unit interval

### Augmentation, training partition only

Rotation plus or minus 20 degrees, width and height shift plus or minus 10
percent, zoom 0.8 to 1.2, horizontal flip. Vertical flip is deliberately
excluded because brain anatomy has a fixed superior to inferior orientation.

### Gaussian Mixture Model stage

Three components, full covariance, maximum 100 iterations, covariance
regularisation 1e-4, fitted by Expectation Maximisation to the intensity
histogram of each enhanced slice.

The lesion component is chosen by a scoring rule that rewards high mean
intensity and penalises components occupying more than 45 percent or less than
0.5 percent of the slice. Selecting the brightest component naively is
unreliable because the skull is frequently the brightest structure.

The mask is refined by morphological opening then closing with a 5 by 5
elliptical kernel, and the largest connected component is retained.

The binary mask is then blurred into a continuous attention map and the
background is attenuated to 25 percent of its original intensity rather than
removed. Hard masking would discard the anatomical context required to
distinguish a meningioma at the brain margin from a pituitary lesion at the
skull base, which differ principally by location.

### Network architecture

Identical for both experiments.

```
Block 1: Conv2D 32 filters 3x3, BatchNormalization, ReLU, MaxPooling 2x2, Dropout 0.25
Block 2: Conv2D 64 filters 3x3, BatchNormalization, ReLU, MaxPooling 2x2, Dropout 0.25
Block 3: Conv2D 128 filters 3x3, BatchNormalization, ReLU, MaxPooling 2x2, Dropout 0.25
Block 4: Conv2D 256 filters 3x3, BatchNormalization, ReLU, MaxPooling 2x2, Dropout 0.25
GlobalAveragePooling2D
Dense 512, BatchNormalization, ReLU, Dropout 0.50
Dense 4, softmax
```

Batch normalisation precedes the activation. Convolutions use He normal
initialisation and no bias, since batch normalisation supplies the shift.

### Training protocol

Adam optimiser, initial learning rate 0.001, halved after 5 epochs without
validation improvement with a floor of 1e-6, batch size 32, maximum 50 epochs,
early stopping with patience 10 on validation accuracy restoring best weights,
categorical cross entropy loss, inverse frequency class weights normalised to
mean one, random seed 42 applied to Python, NumPy and TensorFlow.

### Comparison models

MobileNetV2 transfer learning; and classical baselines consisting of logistic
regression, Gaussian naive Bayes, random forest, support vector machine with a
radial basis kernel, and a stacking ensemble. The classical models consume
mixture descriptors concatenated with the leading 50 principal component scores
of the downscaled grayscale image.

### Evaluation

Weighted F1 is the primary selection criterion. Also reported: accuracy, macro
averaged precision, recall and F1, Cohen's kappa, Matthews correlation
coefficient, per class metrics, one versus rest area under ROC, and confusion
matrices in raw and row normalised form.

False negatives, meaning a tumour bearing scan labelled healthy, are reported
separately from false positives because the costs are asymmetric. The healthy
class index is 2.

Statistical procedures: McNemar's paired test at the five percent level,
bootstrap confidence intervals with 2,000 resamples at 95 percent, and the
intersection over union between Grad-CAM attributions and the segmentation mask.

### Repository layout

```
Brain-Tumor-Detection/
  app/streamlit_app.py
  data/                                   dataset, excluded from version control
  docs/PRD.md, MASTER_PROMPT.md, SETUP.md
  models/                                 generated artefacts
  notebooks/
    01_exploratory_data_analysis.ipynb
    02_gmm_segmentation_and_pca.ipynb
    03_cnn_baseline.ipynb
    04_hybrid_gmm_cnn.ipynb
    05_model_comparison_and_results.ipynb
  results/figures/, results/metrics/
  src/
    config.py, data_loader.py, preprocessing.py, gmm_segmentation.py,
    cnn_model.py, classical_models.py, evaluate.py, explainability.py, train.py
  requirements.txt, README.md, .gitignore
```

### Module interfaces

`src.config`
: `CLASS_FOLDERS`, `CLASS_LABELS`, `CLASS_COLORS`, `CLASS_DESCRIPTIONS`,
  `NUM_CLASSES`, `IMAGE_SIZE`, `IMAGE_SHAPE`, `RANDOM_SEED`, `BATCH_SIZE`,
  `EPOCHS`, `LEARNING_RATE`, `VALIDATION_SPLIT`, `GMM_N_COMPONENTS`,
  `PCA_N_COMPONENTS`, `DATA_DIR`, `TRAIN_DIR`, `TEST_DIR`, `MODELS_DIR`,
  `FIGURES_DIR`, `METRICS_DIR`, `CACHE_DIR`, model filename constants,
  `MEDICAL_DISCLAIMER`

`src.data_loader`
: `verify_dataset`, `list_image_files`, `load_split`, `load_dataset`,
  `stratified_validation_split`, `class_distribution`, `compute_class_weights`,
  `describe_image_properties`

`src.preprocessing`
: `to_grayscale`, `to_uint8`, `apply_clahe`, `denoise`, `resize`, `normalise`,
  `preprocess_image`, `preprocess_batch`, `load_and_preprocess`,
  `build_augmenter`, `make_training_generator`, `summarise_intensity`

`src.gmm_segmentation`
: `fit_gmm_on_image`, `identify_lesion_component`, `clean_mask`,
  `largest_connected_component`, `segment_image`, `colourise_label_map`,
  `apply_attention_mask`, `transform_for_hybrid`, `transform_batch_for_hybrid`,
  `extract_gmm_features`, `extract_gmm_features_batch`, `flatten_for_pca`,
  `fit_pca`, `transform_pca`, `explained_variance_table`

`src.cnn_model`
: `set_global_seed`, `build_cnn`, `build_transfer_model`, `compile_model`,
  `build_callbacks`, `summarise_architecture`

`src.classical_models`
: `MODEL_BUILDERS`, `train_classical_models`, `save_estimator`, `load_estimator`

`src.evaluate`
: `compute_metrics`, `per_class_metrics`, `clinical_error_summary`,
  `plot_confusion_matrix`, `plot_training_history`, `plot_roc_curves`,
  `plot_model_comparison`, `mcnemar_test`, `bootstrap_confidence_interval`,
  `save_metrics`, `load_metrics`, `build_comparison_table`

`src.explainability`
: `find_last_convolutional_layer`, `compute_gradcam_heatmap`,
  `overlay_heatmap`, `generate_explanation`, `heatmap_mask_agreement`

### Command line

```
python -m src.train --experiments baseline hybrid --epochs 50
python -m src.train --experiments baseline --epochs 3 --limit-per-class 200
python -m src.train --experiments baseline hybrid transfer classical
```

### Application

Streamlit, four sections: Overview, Image Analysis, Results, Methodology.
Entry point `app/streamlit_app.py`. Deployed on Streamlit Community Cloud from
GitHub, with model weights hosted externally.

### Course outcome mapping

CO1 end to end pipeline; CO2 exploratory analysis; CO3 attribute importance via
ANOVA, PCA and attribution; CO4 derived transformations; CO5 validated
predictive models; CO6 prescriptive decisions via confidence thresholds and
localisation; CO7 unsupervised pattern discovery; CO8 algorithm comparison and
communication.

### Writing conventions

- Formal technical English, no emojis, no em-dashes or en-dashes
- Report negative results honestly; do not overstate findings
- Every figure and table must be reproducible from the committed code
- The medical disclaimer must appear in the application and the README
- Numbers quoted anywhere must match the computed artefacts in `results/metrics`

## END CONTEXT

---

## Request

Replace this line with the specific task. Examples:

- Add a K equal to 3 versus K equal to 6 ablation for the mixture component count.
- The hybrid model underperforms the baseline by two percentage points. Diagnose
  likely causes in the segmentation stage and propose three concrete experiments.
- Add a calibration analysis using reliability diagrams and expected calibration
  error, as a new section in notebook 05.
- Review `src/gmm_segmentation.py` for correctness and suggest improvements to
  the component selection rule.
- Produce the slide outline for a fifteen minute demonstration of this project.
