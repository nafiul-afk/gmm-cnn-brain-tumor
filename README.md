# Brain Tumour Classification from Magnetic Resonance Imaging

A hybrid machine learning pipeline that combines unsupervised Gaussian Mixture
Model segmentation with supervised convolutional classification to categorise
brain magnetic resonance slices into four diagnostic classes.

Developed for **DS 4889 Machine Learning**, Department of Computer Science and
Engineering, United International University.

---

## Objective

Determine whether unsupervised Gaussian Mixture Model pre-segmentation improves
the accuracy of convolutional brain tumour classification, relative to an
otherwise identical network trained on unsegmented input.

The comparison is controlled: network architecture, random seeds, data
partitions, augmentation policy and training schedule are held constant, and the
only varying factor is the input transformation. The result is reported in
whichever direction it falls.

---

## Diagnostic classes

| Class | Description |
|-------|-------------|
| Glioma | Tumour originating in the glial cells supporting neurons |
| Meningioma | Tumour arising from the meninges covering the brain |
| No Tumor | No abnormal mass detected |
| Pituitary | Tumour forming in the pituitary gland at the skull base |

---

## Models compared

| Identifier | Input | Classifier | Role |
|------------|-------|-----------|------|
| CNN baseline | Preprocessed slice | Four block convolutional network | Reference performance |
| GMM and CNN hybrid | Attention weighted slice | Identical network | Proposed method |
| MobileNetV2 transfer | Preprocessed slice | Pretrained backbone with new head | Capacity comparison |
| Classical baselines | Mixture descriptors and PCA scores | Logistic regression, naive Bayes, random forest, support vector machine, stacking ensemble | Representation learning control |

---

## Pipeline

```
Magnetic resonance slice
          |
          v
   Preprocessing            resize to 224 x 224, CLAHE enhancement, normalisation
          |
          v
   GMM segmentation         three component mixture fitted by Expectation Maximisation,
          |                 lesion component selected under an area constraint,
          |                 binary mask cleaned by morphological opening and closing
          v
   Soft attention           mask blurred into a continuous weight map, background
          |                 attenuated rather than removed to preserve context
          v
   Convolutional network    four blocks with batch normalisation and dropout,
          |                 global average pooling, softmax over four classes
          v
   Output                   class probabilities and Grad-CAM attribution map
```

---

## Repository layout

```
Brain-Tumor-Detection/
  app/
    streamlit_app.py                       Multi page web interface
  data/                                    Dataset, downloaded separately
  docs/
    PRD.md                                 Product requirements document
    MASTER_PROMPT.md                       Reusable project context brief
    SETUP.md                               Environment and dataset instructions
  models/                                  Trained artefacts, generated
  notebooks/
    00_colab_setup.ipynb                   Optional Colab bootstrap for GPU training
    01_exploratory_data_analysis.ipynb
    02_gmm_segmentation_and_pca.ipynb
    03_cnn_baseline.ipynb
    04_hybrid_gmm_cnn.ipynb
    05_model_comparison_and_results.ipynb
  results/
    figures/                               Generated figures
    metrics/                               Generated metric artefacts
  src/
    config.py                              Central configuration
    data_loader.py                         Dataset discovery and splitting
    preprocessing.py                       Enhancement, resizing, augmentation
    gmm_segmentation.py                    Mixture model segmentation and PCA
    cnn_model.py                           Network architectures and callbacks
    classical_models.py                    Engineered feature baselines
    evaluate.py                            Metrics, figures, statistical tests
    explainability.py                      Grad-CAM attribution
    train.py                               Command line training entry point
  requirements.txt
  README.md
```

---

## Getting started

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS and Linux

# Runtime dependencies
pip install -r requirements.txt

# Add the notebook toolchain for local analysis
pip install -r requirements-dev.txt
```

### 2. Obtain the dataset

Download the Brain Tumor MRI Dataset from Kaggle and extract it so that the
following paths exist:

```
data/Training/glioma/
data/Training/meningioma/
data/Training/notumor/
data/Training/pituitary/
data/Testing/glioma/
data/Testing/meningioma/
data/Testing/notumor/
data/Testing/pituitary/
```

Using the Kaggle command line client:

```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data
cd data && unzip brain-tumor-mri-dataset.zip
```

Verify the layout:

```bash
python -c "from src import data_loader; print(data_loader.verify_dataset().to_string(index=False))"
```

### 3. Run the analysis

Execute the notebooks in order:

| Notebook | Produces |
|----------|----------|
| `00_colab_setup.ipynb` | Optional. Prepares a Colab session, downloads the dataset, runs training on a graphics processor |
| `01_exploratory_data_analysis.ipynb` | Class distribution, image geometry, intensity analysis |
| `02_gmm_segmentation_and_pca.ipynb` | Segmentation validation, component selection, PCA study |
| `03_cnn_baseline.ipynb` | Baseline model and metrics |
| `04_hybrid_gmm_cnn.ipynb` | Hybrid model, metrics, attribution overlap analysis |
| `05_model_comparison_and_results.ipynb` | Final tables, figures and significance tests |

Alternatively, reproduce the experiments from the command line:

```bash
# Full protocol
python -m src.train --experiments baseline hybrid --epochs 50

# Fast verification run
python -m src.train --experiments baseline --epochs 3 --limit-per-class 200

# Include the transfer learning and classical comparisons
python -m src.train --experiments baseline hybrid transfer classical

# Point at a dataset held outside the default data folder
python -m src.train --data-dir path/to/dataset --experiments baseline hybrid
```

### 4. Launch the application

```bash
streamlit run app/streamlit_app.py
```

The interface opens at `http://localhost:8501` and provides four sections:
project overview, image analysis, experimental results and methodology.

---

## Deployment

The repository is structured for deployment on Streamlit Community Cloud.

1. Push the repository to GitHub. The `.gitignore` excludes the dataset, the
   cache and model binaries.
2. Host the trained model on Hugging Face Hub or attach it to a GitHub release,
   because model artefacts commonly exceed the file size limits of a repository.
3. Connect the repository at `share.streamlit.io` and set the entry point to
   `app/streamlit_app.py`.

---

## Results

### Dataset characteristics, from notebook 01

| Property | Value |
|----------|-------|
| Total images | 7,200 |
| Class balance | Exactly balanced, ratio 1.00 to 1 |
| Distinct raw resolutions | 91 in a 600 image sample |
| Images already square | 80 percent |
| Channel counts present | 1 and 3, mixed grayscale and colour |
| ANOVA on mean intensity | F equal to 123.4, p below 0.001, eta squared 0.383 |

Class membership explains 38.3 percent of the variance in mean image intensity.
Intensity therefore carries real signal, but it discards all spatial
information, which motivates a model with spatial inductive bias.

### Segmentation study, from notebook 02

| Property | Value |
|----------|-------|
| Component count minimising BIC and AIC | 6 |
| Component count adopted | 3 |
| Variance retained by 50 principal components | 62.5 percent |

The information criteria favour six components, while the project adopts three.
The criterion curve is flat across that region, three components admit a direct
anatomical reading as background, parenchyma and hyperintense tissue, and larger
values fragment the parenchyma into bands that no simple rule can nominate as a
lesion candidate. This is a deliberate trade of likelihood for interpretability
and mask stability, and it is recorded here as a choice rather than an optimum.

Principal component analysis shows overlapping class clouds with displaced
centroids. The leading components encode global factors such as overall
brightness and head size rather than focal pathology, which supports the use of
a convolutional model over a linear one.

### Model comparison

Pending. Populate this section from `results/metrics/final_comparison.csv` once
notebooks 03 to 05 have been executed.

| Model | Accuracy | Precision | Recall | F1 weighted |
|-------|----------|-----------|--------|-------------|
| CNN baseline | | | | |
| GMM and CNN hybrid | | | | |
| MobileNetV2 transfer | | | | |
| Best classical baseline | | | | |

Report the McNemar p value and state the conclusion in whichever direction the
evidence falls.

---

## Evaluation methodology

Performance is reported with weighted and macro averaged precision, recall and
F1 score, Cohen's kappa and the Matthews correlation coefficient. Accuracy alone
is insufficient because the dataset is mildly imbalanced.

Three additional analyses support the conclusions:

- **Clinical error accounting.** False negatives, meaning a tumour bearing scan
  labelled healthy, are reported separately from false positives, because the
  two error types carry very different costs.
- **Bootstrap confidence intervals.** Point estimates are accompanied by
  intervals obtained by resampling the test set, which communicates the
  uncertainty arising from a finite evaluation sample.
- **McNemar's paired test.** Because both models are evaluated on identical
  samples, a paired test is required. The test examines only the discordant
  cases where exactly one model is correct.

---

## Reproducibility

Every stochastic component is seeded from `config.RANDOM_SEED`. The segmentation
stage is deterministic given the seed, and its output is cached to
`cache/hybrid_arrays.npz` so that repeated training runs consume identical
inputs.

---

## Limitations

- A single publicly available dataset was used, so generalisation across
  scanners, acquisition protocols and patient populations is unverified.
- The analysis operates on two dimensional slices rather than full volumes.
- The four category taxonomy is coarse relative to clinical tumour grading.
- No radiologist adjudication or prospective evaluation was performed.
- Predicted probabilities are uncalibrated and should not be interpreted as
  true posterior probabilities.

---

## Disclaimer

This system was developed for academic coursework and research demonstration
only. It is not a certified medical device and must not be used for clinical
diagnosis. All findings require confirmation by a qualified radiologist.

---

## Techniques referenced

| Technique | Where it is used |
|-----------|-----------------|
| Gaussian Mixture Model with Expectation Maximisation | Unsupervised lesion localisation |
| Contrast Limited Adaptive Histogram Equalisation | Preprocessing |
| Convolutional neural network | Primary classifier |
| Batch normalisation and dropout | Regularisation throughout the network |
| Adam optimiser with plateau based scheduling | Training |
| Principal Component Analysis | Classical baseline features and visualisation |
| Random forest, naive Bayes, stacking ensemble | Classical comparison models |
| Gradient weighted Class Activation Mapping | Prediction attribution |
| McNemar's paired test and bootstrap resampling | Statistical comparison |
