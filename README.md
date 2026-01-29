# KSLFN

Yijia Chen, Ze Zhang, Hanwen Zhang, Bingye Zhou, **Zhong Yuan*** (通讯作者), KSLFN: Kernel Sparse Learning with Fuzzy Neighborhood for Outlier Detection, Fuzzy Sets and Systems, 2025.

## Abstract

 Outlier detection focuses on identifying instances that significantly deviate from conventional patterns within large datasets. These outliers often indicate potential risk events such as financial fraud, credit defaults, or cyberattacks. Fuzzy neighborhood outlier detection is effective in many situations. Existing fuzzy neighborhood methods predominantly rely on granular metrics defined within local radii or k-nearest neighbors. While effective for local density estimation, these methods may fail to explicitly capture the underlying global structure that spans the entire dataset. In response to this limitation, this study proposes Kernel Self-representation Learning based Fuzzy Neighborhood outlier detection (KSLFN). Specifically, the kernel self-representation matrix is learned by minimizing the reconstruction error in the high-dimensional feature space, thereby capturing the global structure. Crucially, KSLFN utilizes the pre-computed kernel matrix as a unified foundation: it serves as the computational basis for learning the kernel self-representation matrix while simultaneously characterizing the fuzzy similarity relations for local neighborhood analysis. Finally, the Kernel Self-representation Learning outlier Score and the Fuzzy Neighborhood outlier Score are fused to derive a comprehensive outlier Score, which effectively characterizes the outlier degree of the data. Extensive experiments on public benchmark datasets demonstrate that the proposed method achieves competitive or superior performance compared with existing methods.
## Repository Structure

```
KSLFN/
├── Code/
│   ├── KSLFN.py    # Main algorithm implementation
│   ├── Demo.py     # Demo script
│   └── Example.mat # Small example dataset
├── Datasets/       # 28 benchmark datasets from the paper
└── README.md
```

## Quick Start

Run the demo script in the `Code/` directory:

```bash
cd Code
python Demo.py
```

**Expected Output:**

```
Dataset: Example
Number of samples: 148
Number of features: 18

Running KSLFN with k=3, ksl_weight=0.5...

Outlier scores (higher = more likely to be outlier):
[0.06083036 0.03930366 0.07601323 ... 0.02771192 0.03095525]
```

**Interpretation of Results:**
- Each sample receives an outlier score in `[0, 1]`
- Higher scores indicate samples more likely to be outliers
- Samples with scores > 0.5 (e.g., 0.54, 0.51) are strong outlier candidates

## Usage

```python
from scipy.io import loadmat
from KSLFN import KSLFN

# Load your dataset
mat_data = loadmat("your_dataset.mat")
data_key = next(key for key in mat_data.keys() if not key.startswith("__"))
dataset = mat_data[data_key]

# Prepare data (features normalized to [0,1], last column is label)
trandata = dataset[:, :-1]
labels = dataset[:, -1]

# Run KSLFN
k = 10           # Number of fuzzy k-nearest neighbors
ksl_weight = 0.5 # Weight for KSL component (0.0 to 1.0)
scores = KSLFN(data=trandata, k=k, ksl_weight=ksl_weight)
```

### Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `data` | Input matrix (n_samples × n_features), normalized to [0,1] | Required |
| `k` | Number of fuzzy k-nearest neighbors | 3-50 |
| `ksl_weight` | Weight for KSL score (FN weight = 1 - ksl_weight) | 0.5 |
| `device` | PyTorch device ('cpu', 'cuda', or None) | None |

## Datasets

The `Datasets/` folder contains 28 benchmark datasets used in the paper:

| No. | Dataset | Attributes | Objects | Outliers |
|-----|---------|------------|---------|----------|
| 1 | Abalone | 8 | 4176 | 1 |
| 2 | Audiology | 69 | 226 | 1 |
| 3 | Breast_cancer | 9 | 285 | 85 |
| 4 | Chess_nowin_145 | 36 | 1813 | 145 |
| 5 | Chess_nowin_185 | 36 | 1853 | 185 |
| 6 | German | 20 | 713 | 14 |
| 7 | Ionosphere | 34 | 249 | 24 |
| 8 | Iris | 4 | 110 | 11 |
| 9 | Letter | 32 | 1600 | 100 |
| 10 | Lymphography | 18 | 148 | 2 |
| 11 | Monks_12 | 6 | 240 | 12 |
| 12 | Monks_4 | 6 | 232 | 4 |
| 13 | Mushroom_365 | 22 | 4572 | 364 |
| 14 | Mushroom_467 | 22 | 4674 | 467 |
| 15 | Sick_35 | 29 | 3575 | 35 |
| 16 | Sick_72 | 29 | 3612 | 72 |
| 17 | Sonar | 60 | 106 | 10 |
| 18 | Spambase | 57 | 2843 | 55 |
| 19 | Tic_tac_toe_26 | 9 | 651 | 26 |
| 20 | Tic_tac_toe_32 | 9 | 657 | 32 |
| 21 | Vowels | 12 | 1456 | 50 |
| 22 | Waveform | 21 | 3442 | 99 |
| 23 | Wine | 13 | 129 | 10 |
| 24 | Yeast | 8 | 1141 | 5 |
| 25 | Zoo | 17 | 100 | 4 |

## Requirements

```bash
pip install numpy scipy torch scikit-learn
```

