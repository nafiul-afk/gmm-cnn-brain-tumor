# Project Completion Guide

**Project:** Brain tumour classification from MRI using a hybrid GMM and CNN pipeline
**Repository:** https://github.com/nafiul-afk/gmm-cnn-brain-tumor
**Local path:** `C:\Users\Itachi\OneDrive\Desktop\New folder\Brain-Tumor-Detection`
**Course:** DS 4889 Machine Learning, United International University
**Document written:** 26 October 2026

---

## How to use this document

Work through the phases in order. Each phase states what to do, the exact command
to run, and how to confirm it worked before moving on.

If you are an AI assistant reading this, Section 10 contains the technical
context you need. Read Section 1 first to understand what is already done, so
that you do not repeat completed work.

---

## 1. Current state, verified

### Completed

| Item | Status |
|------|--------|
| Python 3.11.9 and virtual environment at `.venv` | Installed |
| All dependencies | Installed, including TensorFlow 2.21, Keras 3.15, scikit-learn 1.9 |
| Dataset, 7,200 images in `data/` | Downloaded and verified |
| Source package, 10 modules in `src/` | Written and functionally tested, 18 of 18 checks passed |
| Notebook 01, exploratory analysis | Executed, 9 of 9 cells, 5 figures |
| Notebook 02, segmentation and PCA | Executed, 11 of 11 cells, 9 figures |
| Streamlit application | Built, confirmed working |
| Documentation | PRD, setup guide, master prompt, README |
| Git repository, 44 files | Pushed and synced |

### Not completed

| Item | Blocker |
|------|---------|
| Notebook 03, baseline model | Requires training |
| Notebook 04, hybrid model | Requires training |
| Notebook 05, comparison | Requires notebooks 03 and 04 |
| Trained model artefacts | Requires training |
| Metric files | Requires training |
| README results table | Requires metrics |
| Public deployment | Requires a trained model |

### The single outstanding question

The project exists to answer one question:

> Does unsupervised Gaussian Mixture Model pre-segmentation improve convolutional
> brain tumour classification, relative to an identical network trained on
> unsegmented input?

This is currently unanswered because the comparison has never been run.

### Important note on the existing model file

A file `models/best_model.keras` may be present. It is a throwaway demonstration
model trained on 300 images per class for 6 epochs, achieving 38 percent test
accuracy against a 25 percent chance baseline. It exists only to prove the
dashboard works. **Do not report its numbers.** It is overwritten by Phase 2.

---

## 2. Phase 1: Overnight training

This is the critical path. Everything else depends on it.

### Before starting

- Laptop plugged into mains power. On battery Windows will sleep and kill the run.
- At least 10 GB free disk space.
- The terminal window must stay open. Minimising is fine.

### Run it

Open a new PowerShell terminal.

```powershell
cd "C:\Users\Itachi\OneDrive\Desktop\New folder\Brain-Tumor-Detection"
.\run_overnight.ps1
```

If PowerShell refuses to run the script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run_overnight.ps1
```

### What happens

| Stage | Approximate duration |
|-------|---------------------|
| Load and preprocess 7,200 images | 5 minutes |
| Train baseline CNN, up to 50 epochs | 4 hours |
| Apply GMM segmentation to all splits | 30 minutes |
| Train hybrid CNN, up to 50 epochs | 4 hours |
| Evaluate, compare, run McNemar test | 5 minutes |
| **Total** | **6 to 9 hours** |

Early stopping may finish sooner. The model checkpoints after every epoch, so a
crash part way through still leaves usable artefacts.

### Alternative: Google Colab

If you would rather use a GPU and finish in under an hour:

1. Obtain a fresh Kaggle token at kaggle.com/settings, Create New Token.
2. Open this URL:
   `https://colab.research.google.com/github/nafiul-afk/gmm-cnn-brain-tumor/blob/main/notebooks/00_colab_setup.ipynb`
3. Runtime, then Change runtime type, then select T4 GPU, then Save.
4. Run every cell in order. Upload `kaggle.json` when Step 4 prompts.
5. The final cells download `best_model.keras` and `results_bundle.zip`.
6. Place the model in `models/` and unzip the bundle into `results/`.

---

## 3. Phase 2: Check the results

### In the morning

Open the newest file in the `logs` folder named `COMPLETED_<timestamp>.txt`.

It contains the comparison table, the McNemar significance test, and
confirmation that the model was saved.

### Confirm the artefacts exist

```powershell
cd "C:\Users\Itachi\OneDrive\Desktop\New folder\Brain-Tumor-Detection"
Get-ChildItem models\*.keras
Get-ChildItem results\metrics\
```

Expected files:

```
models\best_model.keras
models\cnn_baseline.keras
models\gmm_cnn_hybrid.keras
results\metrics\experiment_comparison.csv
results\metrics\experiment_metrics.json
results\metrics\mcnemar_test.json
```

### Read the outcome

Open `results\metrics\experiment_comparison.csv`. It looks like this:

```
model,accuracy,precision_weighted,recall_weighted,f1_weighted,...
CNN baseline,0.94,0.938,0.94,0.939,...
GMM and CNN hybrid,0.96,0.961,0.96,0.960,...
```

Open `results\metrics\mcnemar_test.json`:

```json
{
  "both_correct": 1180,
  "only_model_a_correct": 42,
  "only_model_b_correct": 78,
  "p_value": 0.0012,
  "significant_at_0.05": true
}
```

### How to interpret it

| Outcome | Meaning |
|---------|---------|
| Hybrid F1 higher and `significant_at_0.05` is true | The hybrid genuinely helps. Hypothesis supported. |
| Hybrid F1 higher but `significant_at_0.05` is false | The difference may be noise. Report it as inconclusive. |
| Baseline F1 higher | The hybrid does not help. **This is a valid result.** Report it honestly. |

A negative result is not a failure. The project is assessed on the rigour of the
comparison, not on the direction of the outcome. If the hybrid loses, say so and
explain why in the limitations.

### If training failed

Open the newest `logs\training_<timestamp>.log` and search for `Traceback` or
`Error`. Section 9 covers common failures.

---

## 4. Phase 3: Run the remaining notebooks

Notebooks 03, 04 and 05 produce the figures and detailed analysis. Notebooks 03
and 04 will retrain unless you point them at the cached artefacts, so budget
time or run them on Colab.

Notebook 05 is the important one and runs quickly, because it only loads the
already trained models.

### Launch the notebook environment

```powershell
cd "C:\Users\Itachi\OneDrive\Desktop\New folder\Brain-Tumor-Detection"
.venv\Scripts\Activate.ps1
jupyter lab
```

Alternatively open the `.ipynb` files directly in Visual Studio Code and select
the kernel named `Python (Brain Tumour)`.

### Run notebook 05 first

`notebooks/05_model_comparison_and_results.ipynb`

It produces:

- Consolidated comparison across all models
- Side by side confusion matrices
- Per class F1 comparison
- Overlaid ROC curves
- McNemar test with a plain English interpretation
- Bootstrap confidence intervals
- Disagreement analysis showing cases where the models differ
- Inference latency comparison

### Then notebooks 03 and 04 if time permits

These regenerate the training curves and the Grad-CAM attribution overlap
analysis. The overlap figure in notebook 04 is worth having, because it tests
whether the network actually attends to the region the GMM identified.

---

## 5. Phase 4: Record the results

Open `README.md` and find the section headed **Model comparison** under
**Results**. It currently contains an empty table.

Fill it from `results/metrics/experiment_comparison.csv`. Replace this:

```markdown
| Model | Accuracy | Precision | Recall | F1 weighted |
|-------|----------|-----------|--------|-------------|
| CNN baseline | | | | |
| GMM and CNN hybrid | | | | |
```

with the real numbers, rounded to three decimal places.

Below the table, add two or three sentences stating:

1. Which model performed better and by how much.
2. The McNemar p value and whether the difference is statistically significant.
3. One sentence of interpretation.

### Rules

- Every number must come from the computed artefacts. Do not estimate or round
  favourably.
- If the hybrid lost, state that plainly.
- Do not claim state of the art. Published figures on this dataset are not
  directly comparable, because preprocessing and test partitions differ.

---

## 6. Phase 5: Commit and push

```powershell
cd "C:\Users\Itachi\OneDrive\Desktop\New folder\Brain-Tumor-Detection"
git add -A
git commit -m "Add training results, metrics and comparison figures"
git push origin main
```

Note that `.gitignore` excludes the dataset, the virtual environment, the cache
and model binaries. The metric files, the figures and the executed notebooks
will be committed, which is intended.

If `git` is not recognised, open a new terminal, or use the full path:

```powershell
& "C:\Program Files\Git\cmd\git.exe" push origin main
```

---

## 7. Phase 6: Deploy the application

### Test locally first

```powershell
cd "C:\Users\Itachi\OneDrive\Desktop\New folder\Brain-Tumor-Detection"
.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

The browser opens at `http://localhost:8501`. Check each section:

- **Overview** shows the dataset composition.
- **Image Analysis** accepts an upload and returns a prediction with confidence
  bars, preprocessing stages, the GMM mask and a Grad-CAM heatmap.
- **Results** displays the metrics table and the McNemar outcome.
- **Methodology** documents the pipeline.

Test by uploading any file from `data\Testing\glioma\` or the other class
folders. With a properly trained model, predictions should be correct most of
the time.

If the loaded model is the hybrid, enable the GMM attention toggle. If it is the
baseline, leave it off.

### Host the model weights

Model files are around 6 MB, which fits within GitHub limits, but they are
currently excluded by `.gitignore`. Two options:

**Option A, simplest.** Remove `models/*.keras` from `.gitignore` and commit the
model. Works because the file is small.

**Option B, cleaner.** Upload `best_model.keras` to a Hugging Face model
repository and modify `app/streamlit_app.py` to download it at startup.

Option A is sufficient for coursework.

### Deploy to Streamlit Community Cloud

1. Go to `share.streamlit.io` and sign in with GitHub.
2. Select **New app**.
3. Repository: `nafiul-afk/gmm-cnn-brain-tumor`
4. Branch: `main`
5. Main file path: `app/streamlit_app.py`
6. Select **Deploy**.

The first build takes several minutes while dependencies install. You receive a
public URL of the form `https://<name>.streamlit.app`.

### Add the URL to the README

Put the live link near the top of `README.md` so anyone opening the repository
can try the application immediately.

---

## 8. Optional extensions

Only consider these once Phases 1 to 7 are complete. A finished project beats an
ambitious unfinished one.

Ranked by value relative to effort:

| Extension | What it adds | Effort |
|-----------|-------------|--------|
| **Sample efficiency curve** | Train both models at 10, 25, 50 and 100 percent of the data. The hypothesis predicts a spatial prior helps most when labels are scarce. If the hybrid wins at 10 percent but ties at 100 percent, this explains the mechanism rather than merely reporting the outcome. | 6 runs |
| **K equal to 3 versus K equal to 6 ablation** | Notebook 02 found that BIC and AIC both favour six mixture components, while the pipeline uses three. This closes the most obvious criticism of the method. | 1 run |
| **Learned attention comparison** | Add squeeze and excitation or CBAM blocks to the baseline. Tests whether a network can learn what the GMM supplies for free. A sharp test of whether the contribution is necessary. | 1 run |
| **Probability calibration** | Reliability diagrams and expected calibration error. Medical predictions should not be read as posterior probabilities unless calibrated. Rarely done in student work. | 1 hour, no training |
| **Semi-supervised learning** | Self-training on 10 percent labelled data. Covers a syllabus topic the project does not currently touch, and addresses the real cost of medical annotation. | 2 runs |
| **Unit tests with pytest** | Engineering credibility for a portfolio. | 2 hours |

If you add only one, choose the sample efficiency curve. It is the only addition
that explains why the method works rather than whether it works.

---

## 9. Troubleshooting

### `python` is not recognised

Windows intercepts the bare `python` command with a Microsoft Store alias. Use
the virtual environment interpreter instead:

```powershell
.venv\Scripts\python.exe -m src.train --help
```

Or activate the environment once per terminal session:

```powershell
.venv\Scripts\Activate.ps1
```

### `git` is not recognised

Git is installed at `C:\Program Files\Git\cmd\git.exe` and registered in the
system PATH. A terminal opened before installation holds a stale environment.
Open a new terminal, or restart Visual Studio Code.

### Training exhausts memory

Reduce the batch size or the amount of data loaded:

```powershell
.venv\Scripts\python.exe -m src.train --experiments baseline hybrid --epochs 50 --batch-size 16
```

Loading the full dataset at 224 by 224 pixels in three channels requires roughly
4 GB.

### The GMM segmentation stage is slow

This is expected. One mixture model is fitted per image. The result is cached to
`cache/hybrid_arrays.npz`, so the cost is paid only once. Do not pass
`--no-cache` unless you intend to recompute it.

### Validation accuracy is far below training accuracy

Seen in the demonstration run: 71 percent training against 40 percent
validation. With only 960 training images and 525,028 parameters this is
overfitting, compounded by batch normalisation running averages that have not
converged after six epochs. Both resolve with the full dataset and 50 epochs. If
it persists after a full run, increase augmentation or dropout.

### The dashboard reports no trained model

The application searches `models/` for `best_model.keras`, then the hybrid,
baseline and transfer artefacts in that order. Confirm at least one exists.

### TensorFlow reports no GPU on Windows

Expected. TensorFlow releases after 2.10 do not provide native Windows GPU
support. Use Google Colab or WSL2 for GPU training.

### Colab cannot find the repository

The GitHub search dialog sometimes fails even for public repositories. Use the
direct URL instead:

```
https://colab.research.google.com/github/nafiul-afk/gmm-cnn-brain-tumor/blob/main/notebooks/00_colab_setup.ipynb
```

Or upload `notebooks/00_colab_setup.ipynb` manually. The notebook clones the
repository itself, so how it was opened does not matter.

---

## 10. Technical context

This section provides what an AI assistant needs in order to help with this
project. The file `docs/MASTER_PROMPT.md` contains a fuller version.

### Dataset

- Brain Tumor MRI Dataset, Masoud Nickparvar, Kaggle
- Identifier `masoudnickparvar/brain-tumor-mri-dataset`
- 7,200 JPEG images, exactly balanced at 1,800 per class
- Training 5,600 images, 1,400 per class
- Testing 1,600 images, 400 per class
- Folder names `glioma`, `meningioma`, `notumor`, `pituitary`
- Display labels `Glioma`, `Meningioma`, `No Tumor`, `Pituitary`
- The healthy class is index 2

Many published descriptions quote 7,023 images with unequal class counts. Those
figures refer to an earlier revision. The counts above were enumerated directly
from the downloaded archive. Because the dataset is balanced, inverse frequency
class weighting is inert and weighted metrics coincide with macro metrics. Both
are still computed so the pipeline remains correct if the publisher revises the
dataset again.

### Method

Preprocessing resizes to 224 by 224 pixels, applies Contrast Limited Adaptive
Histogram Equalisation with clip limit 2.0 and an 8 by 8 tile grid, then
normalises to the unit interval.

The segmentation stage fits a three component Gaussian Mixture Model to the
intensity histogram of each enhanced slice by Expectation Maximisation. The
lesion component is chosen by a scoring rule rewarding high mean intensity and
penalising components occupying more than 45 percent or less than 0.5 percent of
the slice, because selecting the brightest component naively picks the skull.
The mask is refined by morphological opening then closing, and the largest
connected component is retained.

The mask is then blurred into a continuous attention map and the background is
attenuated to 25 percent rather than removed. Hard masking would discard the
anatomical context that distinguishes a meningioma at the brain margin from a
pituitary lesion at the skull base, which differ principally by location.

The classifier is a four block convolutional network with filters 32, 64, 128
and 256. Each block applies convolution, batch normalisation, ReLU activation,
max pooling and dropout at 0.25. Global average pooling precedes a 512 unit
dense layer with dropout at 0.50, then a four way softmax. Total parameters
525,028.

Training uses Adam at an initial learning rate of 0.001, halved after five
epochs without validation improvement, batch size 32, up to 50 epochs, early
stopping with patience 10 restoring the best weights, and categorical cross
entropy loss. The random seed is 42 across Python, NumPy and TensorFlow.

The comparison is controlled: architecture, seeds, partitions, augmentation and
schedule are identical between the two conditions, and only the input
transformation varies.

### Module reference

| Module | Responsibility |
|--------|---------------|
| `src/config.py` | All constants. Change values here, not elsewhere. |
| `src/data_loader.py` | Dataset verification, loading, stratified splitting, class weights |
| `src/preprocessing.py` | CLAHE, resizing, normalisation, augmentation |
| `src/gmm_segmentation.py` | Mixture fitting, component selection, mask refinement, attention weighting, descriptor extraction, PCA |
| `src/cnn_model.py` | Architectures, compilation, callbacks |
| `src/classical_models.py` | Logistic regression, naive Bayes, random forest, SVM, stacking |
| `src/evaluate.py` | Metrics, figures, McNemar test, bootstrap intervals |
| `src/explainability.py` | Grad-CAM computation and attribution overlap |
| `src/train.py` | Command line orchestration |
| `app/streamlit_app.py` | Four section web interface |

### Command reference

```powershell
# Verify the dataset
.venv\Scripts\python.exe -c "from src import data_loader; print(data_loader.verify_dataset().to_string(index=False))"

# Fast smoke test
.venv\Scripts\python.exe -m src.train --experiments baseline --epochs 3 --limit-per-class 100

# Full protocol
.venv\Scripts\python.exe -m src.train --experiments baseline hybrid --epochs 50

# Include transfer learning and classical baselines
.venv\Scripts\python.exe -m src.train --experiments baseline hybrid transfer classical --epochs 50

# Point at a dataset outside the default folder
.venv\Scripts\python.exe -m src.train --data-dir path\to\dataset --experiments baseline

# Launch the application
streamlit run app/streamlit_app.py
```

### Environment versions

Python 3.11.9, TensorFlow 2.21.0, Keras 3.15.1, scikit-learn 1.9.1, NumPy 2.4.6,
pandas 3.1.0, OpenCV 5.0.0, Streamlit 1.66.1.

Two compatibility issues were already resolved and should not be reintroduced:
`LogisticRegression` no longer accepts a `multi_class` argument in scikit-learn
1.7 and later, and the Grad-CAM gradient model requires a fallback path under
Keras 3.

### Writing conventions

- Formal technical English throughout
- No emojis, no em-dashes or en-dashes
- Report results honestly in whichever direction they fall
- Every number quoted must match the computed artefacts in `results/metrics`
- The medical disclaimer must appear in the application and the README

---

## 11. Completion checklist

- [ ] Overnight training completed without error
- [ ] `models/best_model.keras` exists and is not the 38 percent demonstration model
- [ ] `results/metrics/experiment_comparison.csv` contains both models
- [ ] `results/metrics/mcnemar_test.json` exists
- [ ] Notebook 05 executed, comparison figures generated
- [ ] README results table populated from the computed artefacts
- [ ] Interpretation written, including the McNemar outcome
- [ ] Changes committed and pushed to GitHub
- [ ] Application tested locally with the trained model
- [ ] Application deployed to Streamlit Community Cloud
- [ ] Live URL added to the README
- [ ] Kaggle API token rotated, the earlier one was exposed
- [ ] Medical disclaimer present in the application and the README

---

## 12. Outstanding security item

A Kaggle API token was pasted into a chat session during setup and should be
treated as compromised. Rotate it at kaggle.com/settings by selecting Expire
Token and then Create New Token. The credential file currently sits at
`C:\Users\Itachi\.kaggle\kaggle.json`, outside the repository, and was never
committed.

---

## 13. Summary of the immediate next action

Run the overnight training. Everything else depends on it.

```powershell
cd "C:\Users\Itachi\OneDrive\Desktop\New folder\Brain-Tumor-Detection"
.\run_overnight.ps1
```

Keep the laptop plugged in and the terminal open. In the morning, read
`logs\COMPLETED_<timestamp>.txt`.
