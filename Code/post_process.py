"""
Post-processing script: After all experiments are done,
re-select best AUC from w in [0.025, 0.975] only (excluding 0 and 1).
Rewrites KSLFN_results and all_outlier_result_summary.csv.
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.metrics import roc_auc_score

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_base_dir = os.path.join(script_dir, '..', 'Experimental_results')

    # Valid w range: 0.025 to 0.975
    valid_w = [round(w, 3) for w in np.arange(0.025, 0.976, 0.025)]
    print(f"Valid w values ({len(valid_w)}): {valid_w[0]} to {valid_w[-1]}")

    all_metrics = []

    # Find all datasets from any w-folder
    any_w_folder = os.path.join(results_base_dir, f"KSLFN_w_0.025_results")
    if not os.path.exists(any_w_folder):
        print(f"ERROR: Cannot find {any_w_folder}. Make sure experiments finished first.")
        return

    dataset_names = [d for d in os.listdir(any_w_folder)
                     if os.path.isdir(os.path.join(any_w_folder, d))]
    print(f"Found {len(dataset_names)} datasets to re-evaluate.")

    for dataset_name in sorted(dataset_names):
        best_auc = -1
        best_score = None
        best_w = None
        best_k = None

        for w in valid_w:
            w_folder = os.path.join(results_base_dir, f"KSLFN_w_{w:.3f}_results", dataset_name)
            mat_file = os.path.join(w_folder, f"{dataset_name}_KSLFN.mat")

            if not os.path.exists(mat_file):
                continue

            try:
                data = loadmat(mat_file)
                auc = float(np.squeeze(data['auc']))
                if auc > best_auc:
                    best_auc = auc
                    best_score = data['opt_out_scores']
                    best_w = w
                    best_k = int(np.squeeze(data['k']))
            except Exception as e:
                print(f"  Warning: Could not read {mat_file}: {e}")

        if best_score is None:
            print(f"  SKIP {dataset_name}: no valid result found")
            continue

        # Load label from one of the w-files
        sample_file = os.path.join(results_base_dir, f"KSLFN_w_0.025_results",
                                   dataset_name, f"{dataset_name}_KSLFN.mat")
        label = loadmat(sample_file)['label']

        # Save best result to KSLFN_results
        out_dir = os.path.join(results_base_dir, 'KSLFN_results', dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        savemat(os.path.join(out_dir, f"{dataset_name}_KSLFN.mat"), {
            'opt_out_scores': best_score,
            'label': label,
            'auc': best_auc,
            'opt_w': best_w,
            'opt_k': best_k
        })

        # Load KSL and FN only AUCs for summary
        ksl_auc, fn_auc = np.nan, np.nan
        ksl_file = os.path.join(results_base_dir, 'KSL_results', dataset_name, f"{dataset_name}_KSL.mat")
        fn_file = os.path.join(results_base_dir, 'FN_results', dataset_name, f"{dataset_name}_FN.mat")
        if os.path.exists(ksl_file):
            d = loadmat(ksl_file)
            try: ksl_auc = roc_auc_score(d['label'].ravel(), d['opt_out_scores'].ravel())
            except: pass
        if os.path.exists(fn_file):
            d = loadmat(fn_file)
            try: fn_auc = roc_auc_score(d['label'].ravel(), d['opt_out_scores'].ravel())
            except: pass

        all_metrics.append({
            'dataset': dataset_name,
            'KSLFN_best_auc': best_auc,
            'best_w': best_w,
            'best_k': best_k,
            'KSL_only_auc': ksl_auc,
            'FN_only_auc': fn_auc
        })
        print(f"  {dataset_name}: best AUC={best_auc:.4f} (w={best_w:.3f}, k={best_k})")

    # Save summary CSV
    df = pd.DataFrame(all_metrics)
    out_csv = os.path.join(results_base_dir, 'all_outlier_result_summary.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nDone! Summary saved to {out_csv}")

if __name__ == '__main__':
    main()
