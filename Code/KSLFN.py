# Kernel Sparse Learning with Fuzzy Neighborhood (KSLFN) outlier detection.
# Implemented with reference to KSL and Fuzzy Neighborhood literature.

import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import math


DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RIDGE_EPS = 1e-6  # 小幅正则化，缓解求解时的病态矩阵

def _resolve_device(device=None):
    """Return a torch.device, defaulting to auto-detected CPU/GPU."""
    if device is None:
        return DEFAULT_DEVICE
    if isinstance(device, str):
        return torch.device(device)
    return device


def compute_auc(y_true, y_score):
    """Compute and print ROC AUC."""
    auc = roc_auc_score(y_true, y_score)
    print(f"AUC: {auc:.6f}")
    return auc


def plot_roc_curve(y_true, y_score, title="ROC Curve"):
    """Plot ROC curve for given labels and scores."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

#以上代码与算法本身无关
def KSL(data, device=None):
    """Kernel sparse representation that returns affinity matrix and KSL score."""
    thr = 1e-6

    data_np = np.asarray(data)
    n = data_np.shape[0]
    n_nonzero = int(math.sqrt(n))
    # Heuristic: features with values <=1 are treated as numeric (already scaled), others nominal
    num_fea = np.all(data_np <= 1, axis=0)
    nom_fea = ~num_fea

    num_dis = squareform(pdist(data_np[:, num_fea])) if num_fea.any() else np.zeros((n, n))
    nom_dis = squareform(pdist(data_np[:, nom_fea], metric="hamming")) if nom_fea.any() else np.zeros((n, n))
    distance_matrix = num_dis + nom_dis * np.sum(nom_fea)

    gamma = float(np.median(distance_matrix))
    if gamma == 0:
        gamma = 1e-12

    device = _resolve_device(device)
    distance_tensor = torch.from_numpy(distance_matrix).to(device=device, dtype=torch.float64)
    K = torch.exp(-(distance_tensor ** 2) / (2 * (gamma ** 2)))

    coef_matrix = torch.zeros((n, n), device=device, dtype=torch.float64)
    reconstruction_error = torch.zeros(n, device=device, dtype=torch.float64)

    for i in range(n):
        support = []
        coeffs = None
        coherence = torch.abs(K[i]).clone()
        coherence[i] = 0.0
        reconstruction_error[i] = torch.clamp(K[i, i], min=0.0)

        for _ in range(n_nonzero):
            max_idx = int(torch.argmax(coherence).item())
            if coherence[max_idx].item() < thr:
                break

            if max_idx not in support:
                support.append(max_idx)
            support_tensor = torch.tensor(support, device=device, dtype=torch.long)

            k_support = K.index_select(0, support_tensor).index_select(1, support_tensor)
            k_i_support = K[i, support_tensor]

            size = support_tensor.numel()
            identity = torch.eye(size, device=device, dtype=k_support.dtype)
            k_support_reg = k_support + RIDGE_EPS * identity
            coeffs = torch.linalg.solve(k_support_reg, k_i_support)
            reconstruction = torch.sum(
                coeffs[:, None] * K.index_select(0, support_tensor), dim=0
            )
            residual = K[i] - reconstruction
            residual_norm = torch.dot(residual, residual)

            coherence = torch.abs(residual)
            coherence[i] = 0.0
            if support_tensor.numel() > 0:
                coherence.index_fill_(0, support_tensor, 0.0)

            if residual_norm.item() < thr:
                break
        #print(coeffs.shape)
        if support and coeffs is not None:
            support_tensor = torch.tensor(support, device=device, dtype=torch.long)
            coef_matrix[i, support_tensor] = coeffs
            reconstruction_error[i] = torch.clamp(residual_norm, min=0.0)

    KSL_score = torch.sqrt(torch.clamp(reconstruction_error, min=0.0))
    max_score = KSL_score.max()
    if max_score.item() > 0:
        KSL_score = KSL_score / max_score

    KSL_score_np = KSL_score.detach().cpu().numpy()

        # norm = torch.linalg.norm(coef_matrix, dim=1, keepdim=True)
        # norm = torch.where(norm == 0, 1 , norm)
        # normalized_matrix = coef_matrix / norm
        # cosine_similarity = torch.clamp(normalized_matrix @ normalized_matrix.T, min=0.0)
        # cosine_similarity_np = cosine_similarity.detach().cpu().numpy()
    K_np = K.detach().cpu().numpy()

    return K_np, KSL_score_np


def FN(data, k, affinity_matrix=None):
    """Compute fuzzy-neighborhood outlier score using DFNO notation."""
    n = data.shape[0]
    k = int(np.clip(k, 1, n - 1))

    # Use cosine similarity relation from data.
    sim = affinity_matrix

    similarity = np.sort(sim, axis=1)[:, ::-1]
    num = np.argsort(-sim, axis=1, kind="stable")
    ksimilarity = similarity[:, k]
    fkNN_temp = np.where(sim >= ksimilarity[:, None], sim, 0.0)

    fkNN_card = np.sum(fkNN_temp, axis=1)
    count = np.sum(fkNN_temp != 0, axis=1)

    reachsim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            reachsim[i, j] = min(sim[i, j], ksimilarity[j])

    lrd = np.zeros(n)
    for i in range(n):
        if fkNN_card[i] == 0:
            continue
        sum_reachdist = 0.0
        for j in range(count[i]):
            neighbor_idx = num[i, j + 1]
            sum_reachdist += reachsim[i, neighbor_idx]
        if sum_reachdist > 0:
            lrd[i] = sum_reachdist / fkNN_card[i]
    lrd = np.where(lrd > 0, lrd, 1e-11)

    FLDD = np.zeros(n)
    for i in range(n):
        if fkNN_card[i] == 0:
            FLDD[i] = 1.0
            continue
        sumlrd = 0.0
        for j in range(count[i]):
            neighbor_idx = num[i, j + 1]
            sumlrd += lrd[neighbor_idx] / lrd[i]
        FLDD[i] = sumlrd / fkNN_card[i] if lrd[i] > 0 else 1.0

    min_val = np.min(FLDD)
    max_val = np.max(FLDD)
    if max_val > min_val:
        FNOS = (FLDD - min_val) / (max_val - min_val)
    else:
        FNOS = FLDD.copy()

    return FNOS


def KSLFN(data, k, ksl_weight, device=None):
    """Combine KSL and FN scores with configurable weights."""
    ksl_weight = float(np.clip(ksl_weight, 0.0, 1.0))
    fn_weight = 1.0 - ksl_weight

    affinity_matrix, ksl_score = KSL(data, device=device)
    fn_score = FN(
        data,
        k=k,
        affinity_matrix=affinity_matrix 
    )

    combined_score = ksl_weight * ksl_score + fn_weight * fn_score
    return combined_score


if __name__ == "__main__":
    data_path = "KSLFN_demo/ionosphere_b_24_variant1.mat"
    mat_data = loadmat(data_path)
    data_key = next(key for key in mat_data.keys() if not key.startswith("__"))
    dataset = mat_data[data_key]

    trandata = dataset[:, :-1]
    labels = dataset[:, -1]

    device = _resolve_device(None)
    print(f"Using device: {device}")

    scores = KSLFN(
        data=trandata,
        k=50,
        ksl_weight=1,
        device=device
    )

    print(f"Scores: {scores}")
    compute_auc(labels, scores)
   # plot_roc_curve(labels, scores)
