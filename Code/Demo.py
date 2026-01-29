# -*- coding: utf-8 -*-
"""
Demo script for KSLFN: Kernel Sparse Learning with Fuzzy Neighborhood
"""

import numpy as np
from scipy.io import loadmat
from KSLFN import KSLFN

# Load example dataset
mat_data = loadmat("Example.mat")
data_key = next(key for key in mat_data.keys() if not key.startswith("__"))
dataset = mat_data[data_key]

# Prepare data: features and labels
trandata = dataset[:, :-1]  # All columns except last (features, normalized to [0,1])

print(f"Dataset: Example")
print(f"Number of samples: {trandata.shape[0]}")
print(f"Number of features: {trandata.shape[1]}")

# Run KSLFN algorithm
k = 3            # Number of fuzzy k-nearest neighbors
ksl_weight = 0.5 # Weight for KSL component (0.0 to 1.0, FN weight = 1 - ksl_weight)

print(f"\nRunning KSLFN with k={k}, ksl_weight={ksl_weight}...")
scores = KSLFN(data=trandata, k=k, ksl_weight=ksl_weight)

# Display outlier scores
print(f"\nOutlier scores (higher = more likely to be outlier):")
print(scores)
