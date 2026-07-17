"""
Single dataset runner: run only the specified dataset and save results
Usage: python run_single.py satellite
"""
import os
import sys
import glob
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.metrics import roc_auc_score
from KSLFN import KSL, FN, _resolve_device

def run_one(dataset_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, '..', 'Datasets')
    results_base_dir = os.path.join(script_dir, '..', 'Experimental_results')
    f = os.path.join(dataset_dir, f'{dataset_name}.mat')

    if not os.path.exists(f):
        print(f"ERROR: {f} not found!")
        return

    device = _resolve_device(None)
    print(f"Using device: {device}")
    print(f"Running: {dataset_name} ...")

    mat_data = loadmat(f)
    data_key = next(key for key in mat_data.keys() if not key.startswith("__"))
    dataset = mat_data[data_key]
    X = dataset[:, :-1]
    labels = dataset[:, -1].ravel()
    print(f"  Shape: {X.shape}, outlier rate: {labels.mean():.3f}")

    # Compute KSL once
    K_np, ksl_score = KSL(X, device=device)
    ksl_score = np.clip(ksl_score, 0, 1)

    # Compute FN for each k
    k_values = np.arange(20, 51, 1)
    fn_scores = {}
    for k in k_values:
        fn_sc = FN(X, k=int(k), affinity_matrix=K_np)
        fn_scores[k] = np.clip(fn_sc, 0, 1)

    # Best FN only
    best_pure_fn_auc, best_pure_fn_score = -1, None
    for k in k_values:
        auc = roc_auc_score(labels, fn_scores[k])
        if auc > best_pure_fn_auc:
            best_pure_fn_auc = auc
            best_pure_fn_score = fn_scores[k]

    # KSL only
    best_pure_ksl_auc = roc_auc_score(labels, ksl_score)

    # Save KSL and FN results
    ksl_dir = os.path.join(results_base_dir, "KSL_results", dataset_name)
    os.makedirs(ksl_dir, exist_ok=True)
    savemat(os.path.join(ksl_dir, f"{dataset_name}_KSL.mat"), {
        'opt_out_scores': ksl_score.reshape(-1, 1), 'label': labels.reshape(-1, 1)})

    fn_dir = os.path.join(results_base_dir, "FN_results", dataset_name)
    os.makedirs(fn_dir, exist_ok=True)
    savemat(os.path.join(fn_dir, f"{dataset_name}_FN.mat"), {
        'opt_out_scores': best_pure_fn_score.reshape(-1, 1), 'label': labels.reshape(-1, 1)})

    # w sweep: 0.025 to 0.975
    w_values = np.arange(0.025, 0.976, 0.025)
    best_overall_auc, best_overall_score, best_overall_w, best_overall_k = -1, None, None, None

    for w in w_values:
        w = float(w)
        best_auc_for_w, best_score_for_w, best_k_for_w = -1, None, -1
        for k in k_values:
            score = w * ksl_score + (1.0 - w) * fn_scores[k]
            auc = roc_auc_score(labels, score)
            if auc > best_auc_for_w:
                best_auc_for_w, best_score_for_w, best_k_for_w = auc, score, k

        if best_auc_for_w > best_overall_auc:
            best_overall_auc, best_overall_score = best_auc_for_w, best_score_for_w
            best_overall_w, best_overall_k = w, best_k_for_w

        # Save per-w result
        w_dir = os.path.join(results_base_dir, f"KSLFN_w_{w:.3f}_results", dataset_name)
        os.makedirs(w_dir, exist_ok=True)
        savemat(os.path.join(w_dir, f"{dataset_name}_KSLFN.mat"), {
            'opt_out_scores': best_score_for_w.reshape(-1, 1),
            'label': labels.reshape(-1, 1),
            'auc': best_auc_for_w, 'k': best_k_for_w, 'w': w})

    # Save best into KSLFN_results
    kslfn_dir = os.path.join(results_base_dir, "KSLFN_results", dataset_name)
    os.makedirs(kslfn_dir, exist_ok=True)
    savemat(os.path.join(kslfn_dir, f"{dataset_name}_KSLFN.mat"), {
        'opt_out_scores': best_overall_score.reshape(-1, 1),
        'label': labels.reshape(-1, 1),
        'auc': best_overall_auc, 'opt_k': best_overall_k, 'opt_w': best_overall_w})

    print(f"  -> KSL only AUC:     {best_pure_ksl_auc:.4f}")
    print(f"  -> FN only AUC:      {best_pure_fn_auc:.4f}")
    print(f"  -> Best KSLFN AUC:   {best_overall_auc:.4f} (w={best_overall_w:.3f}, k={best_overall_k})")
    print("Done!")

if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'satellite'
    run_one(name)
