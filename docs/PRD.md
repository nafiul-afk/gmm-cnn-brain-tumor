# Product Requirements Document

**Project:** Brain Tumour Classification from Magnetic Resonance Imaging
**Method:** Hybrid Gaussian Mixture Model segmentation with convolutional classification
**Course:** DS 4889 Machine Learning, Summer 2026
**Institution:** United International University, Department of Computer Science and Engineering
**Deliverable type:** Applied machine learning project
**Document version:** 1.1

---

## 1. Purpose and scope

### 1.1 Objective

Deliver an end to end machine learning system that determines whether an
unsupervised intensity model, fitted independently to each magnetic resonance
slice, provides a useful spatial prior to a convolutional classifier. The
deliverable comprises reproducible code, executed analysis notebooks and a
deployed inference application.

### 1.2 Question under investigation

Does unsupervised Gaussian Mixture Model pre-segmentation improve the accuracy
of convolutional brain tumour classification, relative to an otherwise identical
network trained on unsegmented input?

### 1.3 Hypothesis

Supplying a spatial prior reduces the burden on the network of learning where to
attend from labels alone. The hybrid model is therefore expected to achieve
higher weighted F1 score and, in particular, fewer false negatives, than the
baseline.

### 1.4 Falsification criterion

The hypothesis is rejected if the hybrid model does not exceed the baseline by a
margin that is statistically significant under McNemar's paired test at the five
percent level. A negative result is reported as such; it remains a valid
outcome and is treated as a finding rather than a failure. The project is
assessed on the rigour of the comparison, not on the direction of the result.

### 1.5 Out of scope

- Clinical deployment or any form of regulatory submission
- Three dimensional volumetric analysis
- Multi sequence fusion across T1, T2 and FLAIR acquisitions
- Tumour grading, segmentation boundary delineation or survival prediction
- Patient identifiable data of any kind

---

## 2. Dataset

### 2.1 Source

| Attribute | Value |
|-----------|-------|
| Name | Brain Tumor MRI Dataset |
| Publisher | Masoud Nickparvar, Kaggle |
| Identifier | `masoudnickparvar/brain-tumor-mri-dataset` |
| Total images | 7,200 |
| Format | JPEG, variable resolution |
| Licence | Open, educational and research use |

### 2.2 Composition

| Class | Training | Testing | Total | Share |
|-------|----------|---------|-------|-------|
| Glioma | 1,400 | 400 | 1,800 | 25.0 percent |
| Meningioma | 1,400 | 400 | 1,800 | 25.0 percent |
| No Tumor | 1,400 | 400 | 1,800 | 25.0 percent |
| Pituitary | 1,400 | 400 | 1,800 | 25.0 percent |
| **Total** | **5,600** | **1,600** | **7,200** | 100 percent |

The distribution obtained from the Kaggle API on 26 October 2026 is exactly
balanced, with an imbalance ratio of 1.00 to 1.

This differs from the figures of 7,023 images with unequal class counts that
appear in numerous published descriptions of this dataset. Those figures refer
to an earlier revision. The counts above were obtained by direct enumeration of
the downloaded archive and must be re-verified by any party reproducing this
work, because the publisher may revise the dataset again.

**Consequence for the method.** Balance removes a confound that would otherwise
require careful handling. Inverse frequency class weighting remains enabled in
the implementation but is inert, since every weight evaluates to 1.0. Weighted
and macro averaged metrics coincide for the same reason. Both are nonetheless
reported, so that the analysis remains correct if a future revision of the
dataset reintroduces imbalance.

### 2.3 Partitioning

| Partition | Derivation | Size | Permitted use |
|-----------|-----------|------|---------------|
| Training | 80 percent of the Training folder, stratified | 4,480 | Gradient updates |
| Validation | 20 percent of the Training folder, stratified | 1,120 | Early stopping, learning rate scheduling, model selection |
| Test | The Testing folder, untouched | 1,600 | Final reporting only, evaluated once per model |

Stratification preserves class proportions in both derived subsets.

---

## 3. Method

### 3.1 Preprocessing

| Step | Parameters | Justification |
|------|-----------|---------------|
| Resize | 224 by 224 pixels, area interpolation | Uniform network input; matches common pretrained backbones |
| CLAHE | Clip limit 2.0, tile grid 8 by 8 | Raises local contrast at lesion boundaries without amplifying global noise |
| Normalisation | Divide by 255 | Places inputs in the unit interval for stable gradient magnitudes |

### 3.2 Augmentation

Applied to the training partition only.

| Transformation | Range | Justification |
|----------------|-------|---------------|
| Rotation | plus or minus 20 degrees | Reflects plausible variation in patient head positioning |
| Width and height shift | plus or minus 10 percent | Reflects variation in field of view centring |
| Zoom | 0.8 to 1.2 | Reflects variation in acquisition scale |
| Horizontal flip | Enabled | Brain anatomy is approximately bilaterally symmetric |

Vertical flipping is deliberately excluded, because the superior to inferior
orientation of brain anatomy is fixed and an inverted slice does not represent a
realistic acquisition.

### 3.3 Gaussian Mixture Model segmentation

The intensity histogram of each enhanced slice is modelled as a weighted sum of
Gaussian densities and the parameters are estimated by Expectation Maximisation.

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Components | 3 | Approximates background, normal parenchyma and hyperintense tissue |
| Covariance structure | Full | Permits flexible component widths |
| Maximum iterations | 100 | Sufficient for convergence on one dimensional data |
| Covariance regularisation | 1e-4 | Prevents degenerate components collapsing onto single intensities |

**Component selection.** Selecting the brightest component naively is unreliable,
because the skull and scanner artefacts are frequently the brightest structures.
A scoring rule is applied that rewards high mean intensity and penalises
components occupying more than 45 percent or less than 0.5 percent of the slice.

**Mask refinement.** Morphological opening removes isolated speckle and closing
fills interior gaps. The largest connected component is retained.

**Soft attention.** The binary mask is blurred with a Gaussian kernel into a
continuous weight map, and the background is attenuated to 25 percent of its
original intensity rather than removed. Hard masking would discard the
anatomical context required to distinguish a meningioma at the brain margin from
a pituitary lesion at the skull base, which are differentiated principally by
location rather than by appearance.

### 3.4 Network architecture

Identical for the baseline and the hybrid experiment, so that any performance
difference is attributable to the input transformation alone.

| Stage | Composition | Purpose |
|-------|------------|---------|
| Block 1 | Conv2D 32 filters 3 by 3, batch normalisation, ReLU, max pooling, dropout 0.25 | Edge and texture primitives |
| Block 2 | Conv2D 64 filters, same pattern | Local shape composition |
| Block 3 | Conv2D 128 filters, same pattern | Regional structure |
| Block 4 | Conv2D 256 filters, same pattern | Lesion level abstraction |
| Pooling | Global average pooling | Parameter reduction and structural regularisation |
| Dense | 512 units, batch normalisation, ReLU, dropout 0.50 | Decision boundary in feature space |
| Output | 4 units, softmax | Probability distribution over classes |

Batch normalisation precedes the activation so that the input to the non
linearity has a stable distribution across mini batches, which permits a higher
learning rate. Global average pooling replaces flattening because it reduces the
parameter count by roughly two orders of magnitude while acting as a structural
regulariser.

### 3.5 Training protocol

| Setting | Value |
|---------|-------|
| Optimiser | Adam |
| Initial learning rate | 0.001 |
| Learning rate schedule | Halved after 5 epochs without validation improvement, floor 1e-6 |
| Batch size | 32 |
| Maximum epochs | 50 |
| Early stopping | Patience 10 on validation accuracy, best weights restored |
| Loss | Categorical cross entropy |
| Class weighting | Inverse frequency, mean normalised |
| Random seed | 42, applied to Python, NumPy and TensorFlow |

### 3.6 Comparison models

| Model | Purpose |
|-------|---------|
| MobileNetV2 transfer | Establishes whether representations learned on natural images transfer, and provides a contingency if the custom network underperforms |
| Logistic regression | Linear decision boundary on engineered descriptors |
| Gaussian naive Bayes | Probabilistic reference model from the course text |
| Random forest | Bagging ensemble over decision trees |
| Support vector machine | Non linear kernel boundary |
| Stacking ensemble | Learns how far to trust each base estimator |

The classical models consume a concatenation of mixture model descriptors,
specifically sorted component means, weights and standard deviations, together
with lesion shape statistics, and the leading 50 principal component scores of
the downscaled grayscale image.

---

## 4. Evaluation

### 4.1 Primary metrics

| Metric | Definition | Role |
|--------|-----------|------|
| Weighted F1 | Support weighted harmonic mean of precision and recall | Primary model selection criterion |
| Accuracy | Proportion of correct predictions | Headline figure, reported with the caveat of class imbalance |
| Macro F1 | Unweighted mean across classes | Ensures small class performance is visible |
| Cohen's kappa | Agreement corrected for chance | Guards against inflated agreement |
| Matthews correlation | Balanced correlation coefficient | Robust single number summary |
| Area under ROC | One versus rest, per class | Class separability independent of threshold |

### 4.2 Clinical error accounting

False negatives, meaning a tumour bearing scan labelled as healthy, are reported
separately from false positives. The two error types carry asymmetric costs: a
false negative delays treatment, whereas a false positive causes unnecessary but
non harmful follow up.

### 4.3 Statistical procedures

| Procedure | Purpose |
|-----------|---------|
| McNemar's paired test | Determines whether the difference between two models on identical samples is significant |
| Bootstrap confidence intervals | Communicates uncertainty arising from a finite test set, 2,000 resamples at 95 percent |
| Grad-CAM to mask overlap | Measures intersection over union between network attribution and the segmentation mask, providing direct evidence for or against the design hypothesis |

### 4.4 Success criteria

| Criterion | Minimum | Target |
|-----------|---------|--------|
| Baseline accuracy | 0.90 | 0.94 |
| Hybrid accuracy | 0.92 | 0.96 |
| Hybrid improvement over baseline | Any direction, reported honestly | Positive and significant at p less than 0.05 |
| Hybrid false negative rate | Below baseline | At least 20 percent relative reduction |
| Inference latency per image | Under 2,000 milliseconds | Under 500 milliseconds |
| Notebooks execute end to end | Required | Required |
| Application deployed publicly | Required | Required |
| Result reported in whichever direction it falls | Required | Required |

---

## 5. Deliverables

### 5.1 Software

| Artefact | Description |
|----------|-------------|
| `src/config.py` | Single source of truth for all constants |
| `src/data_loader.py` | Dataset verification, loading, stratified splitting, class weighting |
| `src/preprocessing.py` | CLAHE, resizing, normalisation, augmentation |
| `src/gmm_segmentation.py` | Mixture fitting, component selection, mask refinement, attention weighting, descriptor extraction, PCA |
| `src/cnn_model.py` | Architectures, compilation, callbacks |
| `src/classical_models.py` | Engineered feature baselines and stacking ensemble |
| `src/evaluate.py` | Metrics, figures, McNemar's test, bootstrap intervals |
| `src/explainability.py` | Grad-CAM computation, overlay rendering, mask agreement |
| `src/train.py` | Command line orchestration of the full protocol |
| `app/streamlit_app.py` | Four section web interface |

### 5.2 Analysis

| Notebook | Scope |
|----------|-------|
| 01 | Exploratory data analysis: distribution, geometry, intensity statistics, ANOVA on class separability |
| 02 | Segmentation study: mixture fitting, component count selection by BIC, lesion component identification, PCA analysis |
| 03 | Baseline convolutional model: training, evaluation, error analysis |
| 04 | Hybrid pipeline: segmentation application, controlled training, attribution overlap analysis |
| 05 | Comparative evaluation: consolidated tables, significance tests, disagreement analysis, efficiency comparison |

### 5.3 Generated outputs

| Artefact | Produced by | Location |
|----------|------------|----------|
| Exploratory figures | Notebook 01 | `results/figures` |
| Segmentation and PCA figures | Notebook 02 | `results/figures` |
| Trained model artefacts | Notebooks 03 and 04, or `src.train` | `models` |
| Per model metric files | Notebooks 03 to 05, or `src.train` | `results/metrics` |
| Consolidated comparison table | Notebook 05 | `results/metrics/final_comparison.csv` |
| Paired significance test | Notebook 05, or `src.train` | `results/metrics/mcnemar_test.json` |
| Deployed application | Streamlit Community Cloud | Public URL |

---

## 6. Course outcome mapping

| Outcome | Requirement | Evidence in this project |
|---------|------------|-------------------------|
| CO1 | End to end analysis including framing, acquisition, cleaning and integration | `data_loader.py`, `preprocessing.py`, dataset verification, the complete pipeline from download to deployment |
| CO2 | Exploratory analysis of distributions, relationships and quality issues | Notebook 01: class distribution, image geometry, intensity distributions, one way ANOVA |
| CO3 | Attribute importance via statistical tests and feature selection measures | ANOVA on intensity, PCA explained variance, Grad-CAM attribution, random forest importances |
| CO4 | Derived attributes and transformations | CLAHE enhancement, normalisation, mixture descriptors, attention weighting, PCA scores |
| CO5 | Predictive models with sound validation and interpretation | Stratified three way split, early stopping on validation, test evaluated once, bootstrap intervals |
| CO6 | Prescriptive analytics translating predictions into decisions | Confidence thresholding with an explicit low confidence warning, Grad-CAM localisation, asymmetric error reporting |
| CO7 | Pattern discovery using data mining methodology | Unsupervised mixture segmentation, PCA structure analysis, lesion area ratio differences across classes |
| CO8 | Algorithm selection by comparison on scalability, interpretability, ethics and performance, communicated to technical and non technical audiences | Four way model comparison, latency and parameter accounting, limitations section, Streamlit interface for non technical readers |

---

## 7. Syllabus coverage

| Lecture or topic | Application |
|------------------|-------------|
| 01 Introduction, learning types | Supervised classifier, unsupervised segmentation, discussion of generalisation |
| 02 Supervised fundamentals, evaluation, cross validation, hyperparameter tuning | Stratified splits, metric suite, learning rate scheduling |
| 03 Classification basics, logistic regression, decision boundaries | Logistic regression baseline, softmax output layer |
| 05 Maximum likelihood estimation, MAP versus MLE | Mixture parameters estimated by maximum likelihood |
| 06 Gaussian Mixture Model, Expectation Maximisation | Core of the segmentation stage |
| 08 Neural network fundamentals, activations, backpropagation | Network construction and Grad-CAM gradient computation |
| 09 Gradient descent variants, batch normalisation, dropout, initialisation, optimisers | Adam optimiser, He initialisation, batch normalisation and dropout throughout |
| 11 Convolutional neural networks | Primary classification model |
| Course content: naive Bayes | Gaussian naive Bayes baseline |
| Course content: ensemble methods, bagging, boosting, stacking | Random forest and stacking ensemble baselines |
| Course content: random forest classification | Engineered feature baseline |
| Course content: principal component analysis | Dimensionality reduction for classical models and visualisation |
| Course content: practical applications | Deployed inference application |

---

## 8. Technology

| Layer | Selection | Rationale |
|-------|----------|-----------|
| Language | Python 3.10 or later | Standard for the ecosystem |
| Deep learning | TensorFlow and Keras 2.13 or later | Mature Grad-CAM support via GradientTape |
| Classical learning | scikit-learn 1.3 or later | Reference implementations of GMM, PCA and the ensemble methods |
| Image processing | OpenCV | CLAHE, morphology, connected components |
| Statistics | SciPy and statsmodels | ANOVA and McNemar's test |
| Static figures | Matplotlib and seaborn | High resolution output for reports and slides |
| Interactive figures | Plotly | Interactive charts within the application |
| Application | Streamlit 1.28 or later | Rapid multi page interface with caching primitives |
| Hosting | GitHub with Streamlit Community Cloud | Free, integrates directly with the repository |
| Model storage | Hugging Face Hub or GitHub releases | Model artefacts exceed repository file size limits |

---

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Hybrid does not outperform the baseline | Medium | Medium | The study is framed as a question rather than a claim; a negative result with the attribution overlap analysis remains publishable coursework |
| Segmentation selects the skull rather than a lesion | Medium | High | Area constrained scoring rule, morphological refinement, largest component retention, visual validation in notebook 02 |
| Training exceeds available compute | Medium | Medium | Colab GPU route documented, checkpointing after every epoch, `--limit-per-class` for reduced runs |
| Segmentation stage dominates runtime | High | Low | Output cached to `cache/hybrid_arrays.npz`, computed once |
| Overfitting to a single dataset | High | Medium | Stated explicitly as a limitation; augmentation, dropout and early stopping applied |
| Model artefact exceeds the GitHub file limit | Medium | Low | External hosting documented in the setup guide |
| Misinterpretation as a clinical tool | Low | High | Disclaimer rendered on every application page and in every document |

---

## 10. Schedule

| Week | Phase | Output |
|------|-------|--------|
| 1 | Environment and data acquisition | Verified dataset, notebook 01 |
| 2 | Segmentation study | Notebook 02, validated mixture parameters |
| 3 | Baseline model | Notebook 03, baseline metrics |
| 4 | Hybrid model | Notebook 04, hybrid metrics |
| 5 | Comparative evaluation | Notebook 05, final tables and figures |
| 6 | Application | Streamlit interface, local verification |
| 7 | Deployment | Public repository and hosted application |
| 8 | Consolidation | Final figures, metric artefacts, README results section |
| 9 | Presentation | Demonstration walkthrough and summary slides |

---

## 11. Acceptance checklist

- [ ] Dataset verification reports eight non empty class folders
- [ ] All five notebooks execute from a clean kernel without error
- [ ] Baseline and hybrid models trained with identical seeds and partitions
- [ ] Test partition evaluated exactly once per model
- [ ] McNemar's test executed and reported regardless of its direction
- [ ] Bootstrap confidence intervals reported for both headline metrics
- [ ] Grad-CAM attributions generated and overlap with masks quantified
- [ ] Streamlit application runs locally and handles a missing model gracefully
- [ ] Repository public, dataset and binaries excluded by `.gitignore`
- [ ] Application deployed and reachable by URL
- [ ] README results section populated from computed artefacts, not estimates
- [ ] Medical disclaimer present in the application and the README
- [ ] Limitations stated without overstating the findings
