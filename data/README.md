# Dataset directory

The dataset is not committed to version control. Download it before running any
notebook or training command.

## Source

Brain Tumor MRI Dataset, published by Masoud Nickparvar on Kaggle.

Identifier: `masoudnickparvar/brain-tumor-mri-dataset`

## Download

Using the Kaggle command line client:

```bash
pip install kaggle
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data
unzip data/brain-tumor-mri-dataset.zip -d data
rm data/brain-tumor-mri-dataset.zip
```

Alternatively, download the archive from the dataset page in a browser and
extract it into this directory.

## Required layout

```
data/
  Training/
    glioma/          1,400 images
    meningioma/      1,400 images
    notumor/         1,400 images
    pituitary/       1,400 images
  Testing/
    glioma/            400 images
    meningioma/        400 images
    notumor/           400 images
    pituitary/         400 images
```

Total 7,200 images, distributed evenly across the four classes.

Note that several published descriptions of this dataset quote 7,023 images with
unequal class counts. Those figures refer to an earlier revision. Always treat
the output of the verification command below as authoritative for your copy.

A common extraction error places the content inside a nested folder. Confirm
that `data/Training` exists directly, rather than
`data/brain-tumor-mri-dataset/Training`.

## Verification

```bash
python -c "from src import data_loader; print(data_loader.verify_dataset().to_string(index=False))"
```

The command prints eight rows. Every row must report a non zero image count and
`exists` equal to `True`.
