# Environment and Dataset Setup

This document covers everything required to run the project from a clean
machine. Two routes are described: a local installation and Google Colab. Colab
is recommended if a dedicated graphics processor is not available locally.

---

## Route A: Google Colab (recommended for training)

Training the convolutional models on a central processor takes several hours.
Colab provides a free graphics processor that reduces this to roughly twenty
minutes per model.

> **Shortcut.** The notebook `notebooks/00_colab_setup.ipynb` automates every
> step in this section. Upload it to Colab and execute the cells in order. The
> manual instructions below are retained for reference and for troubleshooting.

### Step 1. Create a notebook and select a GPU runtime

In Colab, open `Runtime`, then `Change runtime type`, and set the hardware
accelerator to `T4 GPU`.

### Step 2. Obtain a Kaggle API token

1. Sign in at `kaggle.com`.
2. Open `Settings`, scroll to the API section and select `Create New Token`.
3. A file named `kaggle.json` is downloaded.

### Step 3. Prepare the environment

Run the following in the first Colab cell.

```python
from google.colab import files
files.upload()          # select kaggle.json when prompted

import os
os.makedirs('/root/.kaggle', exist_ok=True)
os.rename('kaggle.json', '/root/.kaggle/kaggle.json')
os.chmod('/root/.kaggle/kaggle.json', 0o600)
```

### Step 4. Clone the project and download the dataset

```python
!git clone https://github.com/<your-username>/brain-tumor-detection.git
%cd brain-tumor-detection

!pip install -q -r requirements.txt

!kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data
!unzip -q data/brain-tumor-mri-dataset.zip -d data
!rm data/brain-tumor-mri-dataset.zip
```

### Step 5. Verify the layout

```python
from src import data_loader
print(data_loader.verify_dataset().to_string(index=False))
```

The expected output lists eight rows with non zero image counts.

### Step 6. Train

```python
!python -m src.train --experiments baseline hybrid --epochs 50
```

### Step 7. Retrieve the artefacts

```python
from google.colab import files
files.download('models/best_model.keras')
```

---

## Route B: Local installation

### Step 1. Install Python

The project requires Python 3.10 or later. On Windows, install from
`python.org` rather than relying on the Microsoft Store alias, and tick
`Add Python to PATH` during installation.

Verify the installation:

```bash
python --version
```

### Step 2. Create an isolated environment

```bash
cd Brain-Tumor-Detection

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate.bat

# macOS and Linux
source .venv/bin/activate
```

### Step 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If a compatible graphics processor is present on Linux or Windows with WSL,
install the accelerated build instead:

```bash
pip install tensorflow[and-cuda]
```

### Step 4. Download the dataset

Either use the Kaggle client:

```bash
pip install kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data
```

Or download the archive manually from the dataset page and extract it into the
`data` directory.

The resulting structure must be:

```
data/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

### Step 5. Verify

```bash
python -c "from src import data_loader; print(data_loader.verify_dataset().to_string(index=False))"
```

### Step 6. Run a fast smoke test

Before committing to a full training run, confirm that the pipeline executes end
to end on a small subset:

```bash
python -m src.train --experiments baseline --epochs 3 --limit-per-class 100
```

This completes in a few minutes and writes artefacts to `models` and `results`.

### Step 7. Run the full protocol

```bash
python -m src.train --experiments baseline hybrid --epochs 50
```

---

## Running the notebooks

Install the notebook toolchain, which is kept separate from the runtime
dependencies so that a deployment of the application does not pull it in:

```bash
pip install -r requirements-dev.txt
```

Then register the environment as a named kernel:

```bash
python -m ipykernel install --user --name brain-tumour --display-name "Python (Brain Tumour)"
```

Two interfaces are available.

### Visual Studio Code

Open any file in the `notebooks` folder. Select the kernel from the picker in
the upper right of the editor, choosing either the interpreter at
`.venv\Scripts\python.exe` or the registered kernel named
`Python (Brain Tumour)`.

This route requires the Python and Jupyter extensions, both published by
Microsoft.

### JupyterLab

```bash
jupyter lab
```

A browser tab opens at `http://localhost:8888`. Navigate to the `notebooks`
folder and select the `Python (Brain Tumour)` kernel.

### Execution order

Run the notebooks in numerical order. Each one assumes its working directory is
`notebooks` and inserts the project root onto the import path with
`sys.path.append('..')`.

Notebooks 01 and 02 perform exploratory analysis and the segmentation study.
Neither requires a graphics processor and both complete in a few minutes on a
central processor. Notebooks 03 and 04 train the networks and are better run on
Colab.

---

## Running the application

```bash
streamlit run app/streamlit_app.py
```

The application requires at least one trained model in the `models` directory.
If none is present, the Image Analysis page displays instructions rather than
failing.

---

## Expected runtimes

| Task | Central processor | Graphics processor |
|------|------------------|-------------------|
| Dataset loading and preprocessing | 3 to 6 minutes | identical |
| Gaussian Mixture segmentation of all splits | 25 to 40 minutes | identical |
| Baseline training, 50 epochs | 4 to 7 hours | 15 to 25 minutes |
| Hybrid training, 50 epochs | 4 to 7 hours | 15 to 25 minutes |
| Evaluation and figure generation | 2 to 5 minutes | identical |

The segmentation stage does not benefit from a graphics processor because it is
driven by scikit-learn on the central processor. Its output is cached to
`cache/hybrid_arrays.npz` so that the cost is incurred only once.

---

## Troubleshooting

**The dataset verification raises FileNotFoundError.**
The archive extracted into a nested folder. Confirm that `data/Training` exists
directly, rather than `data/brain-tumor-mri-dataset/Training`.

**Training exhausts available memory.**
Reduce the batch size with `--batch-size 16`, or load a subset with
`--limit-per-class 500`. Loading the full dataset at 224 by 224 pixels in three
channels requires approximately 4 gigabytes of memory.

**The segmentation stage is slow.**
This is expected, since one mixture model is fitted per image. Confirm that
caching is enabled by not passing `--no-cache`, so the cost is paid once.

**TensorFlow does not detect the graphics processor.**
Check with `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.
On native Windows, TensorFlow releases after 2.10 do not include graphics
support; use WSL2 or Colab instead.

**Streamlit reports that no model was found.**
Confirm that a file with the extension `.keras` exists in `models`. The
application searches for `best_model.keras`, then the hybrid, baseline and
transfer artefacts in that order.
