# PCA — Handwritten Digit Recognition

> Dimensionality reduction with Principal Component Analysis on the sklearn Digits dataset — compressing 64 pixel features down to a fraction of that size while retaining classification accuracy, with a visual proof of the information trade-off through image reconstruction. 

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Visuals](#visuals)
- [Repo Structure](#repo-structure)
- [Setup](#setup)
- [Tech Stack](#tech-stack)

## Overview

Each handwritten digit image is a flattened 8×8 grid of pixel intensities — 64 features, many of them redundant (neighboring pixels light up together, corner pixels are almost always background). This project uses PCA to compress that redundancy away, and checks what it costs — or doesn't cost — in classification accuracy, while visually confirming *why* PCA works through direct image reconstruction at increasing numbers of components.

## Dataset

**sklearn Digits dataset** (`sklearn.datasets.load_digits`)

| Property | Value |
|---|---|
| Samples | 1,797 |
| Classes | 10 (digits 0–9), balanced |
| Features | 64 (8×8 grayscale pixels, intensity 0–16) |
| Missing values | None |
| Train / Test split | 1,437 / 360 (80/20, stratified) |

No external download required — the dataset ships with scikit-learn.

## Approach

The notebook is organized into 5 modules:

1. **Setup & Data Understanding (EDA)** — load the data, check class balance, view example images, visualize pixel correlation.
2. **Baseline model — no PCA** — standardize all 64 features, train a multinomial Logistic Regression, record accuracy and training time as the benchmark.
3. **Applying PCA** — fit PCA on the standardized training data (`PCA(n_components=0.95)` — sklearn automatically finds the minimum number of components needed for 95% variance), inspect the scree plot and cumulative explained variance.
4. **Model on PCA-reduced data + visualization** — retrain the same classifier on the reduced features, compare to the baseline, visualize the 2D PC1-vs-PC2 projection, and reconstruct sample digits from PCA at increasing component counts.
5. **Insights** — side-by-side comparison table and discussion of the accuracy/compression trade-off.

## Results

| Metric | Baseline (64 features) | PCA-reduced (40 components) |
|---|---|---|
| Accuracy | 97.2% | 95.3% |
| Training time | 0.024 sec | 0.019 sec |
| Dimensionality | 64 | 40 (37.5% reduction) |

PCA compressed the feature space by **~38%** while retaining **95% of the total variance**, at a cost of about **2 points of accuracy**.

## Visuals

**Sample digits from the dataset:**

![Sample digits](images/sample_digits.png)

**Scree plot and cumulative explained variance — the basis for choosing k=40 components:**

![Scree plot and cumulative variance](../PCA-Digits-Dimensionality-Reduction/images/Scree%20plot_and_cumulative_explained_variance.png)

**Reconstruction quality at increasing numbers of components** — the clearest, most intuitive proof of PCA's information trade-off. At k=5 the digits are blurry but recognizable; by k=40 they're nearly indistinguishable from the originals:

![Reconstruction comparison](images/reconstruction_comparison.png)

## Repo Structure

```
├── PCA_Digits_Recognition.ipynb   # full project notebook (all 5 modules, executed)
├── assets/
│   ├── sample_digits.png
│   ├── scree_cumulative_variance.png
│   └── reconstruction_comparison.png
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook PCA_Digits_Recognition.ipynb
```

## Tech Stack

Python · pandas · NumPy · scikit-learn (`PCA`, `StandardScaler`, `LogisticRegression`) · matplotlib · seaborn
