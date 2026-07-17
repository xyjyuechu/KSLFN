import os
import glob
import pandas as pd
import numpy as np
from scipy.io import loadmat, savemat
from sklearn.metrics import roc_auc_score
from KSLFN import KSLFN, KSL, FN, _resolve_device

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, '..', 'Datasets')
    results_base_dir = os.path.join(script_dir, '..', 'Experimental_results')
    
    mat_files = glob.glob(os.path.join(dataset_dir, '*.mat'))
    
    # Check that datasets are 32
    print(f"Total datasets found: {len(mat_files)}")
    
    device = _resolve_device(None)
    print(f"Using device: {device}")
    
    # Setup w values according to paper: [0.0, 0.025, 0.05, ..., 1.0]
    # W=0 means FN, w=1 means KSL, making it easy to just run this single loop!
    w_values = np.arange(0.025, 0.976, 0.025)  # w: 0.025 to 0.975, exclude 0 and 1
    
    # Paper setup: k from 20 to 50
    k_values = np.arange(20, 51, 1)
    
    # Store overview metrics
    all_metrics = []

    for idx, f in enumerate(sorted(mat_files)):
        dataset_name = os.path.basename(f).replace('.mat', '')
        print(f"[{idx+1}/{len(mat_files)}] Evaluating {dataset_name}...")
        
        try:
            # 1. Load data
            mat_data = loadmat(f)
            data_key = next(key for key in mat_data.keys() if not key.startswith("__"))
            dataset = mat_data[data_key]
            
            X = dataset[:, :-1]
            labels = dataset[:, -1].ravel()
            
            # Removed StandardScaler because original KSL code strictly classifies features > 1 as nominal and uses hamming distance
            # The datasets inside .mat are generally pre-scaled to [0,1].
            X_input = X

            # 2. Compute KSL (Independent of k, most computationally heavy, compute ONCE)
            K_np, ksl_score = KSL(X_input, device=device)
            # Clip between [0,1]
            ksl_score = np.clip(ksl_score, 0, 1)

            # 3. For each k, compute FN
            fn_scores = {}
            for k in k_values:
                # FN is very fast since K_np is pre-computed
                fn_sc = FN(X_input, k=int(k), affinity_matrix=K_np)
                fn_scores[k] = np.clip(fn_sc, 0, 1)

            # 4. Search and save for each w
            best_overall_auc = -1
            best_overall_score = None
            best_overall_w = -1
            best_overall_k = -1
            
            best_w_records = []
            
            # Record scores for pure KSL and pure FN separately if required
            best_pure_ksl_auc = roc_auc_score(labels, ksl_score)
            
            best_pure_fn_auc = -1
            best_pure_fn_score = None
            for k in k_values:
                auc_fn = roc_auc_score(labels, fn_scores[k])
                if auc_fn > best_pure_fn_auc:
                    best_pure_fn_auc = auc_fn
                    best_pure_fn_score = fn_scores[k]

            # Save isolated KSL and FN records to their specific folders
            ksl_dir = os.path.join(results_base_dir, "KSL_results", dataset_name)
            os.makedirs(ksl_dir, exist_ok=True)
            savemat(os.path.join(ksl_dir, f"{dataset_name}_KSL.mat"), {
                'opt_out_scores': ksl_score.reshape(-1, 1),
                'label': labels.reshape(-1, 1)
            })
            
            fn_dir = os.path.join(results_base_dir, "FN_results", dataset_name)
            os.makedirs(fn_dir, exist_ok=True)
            savemat(os.path.join(fn_dir, f"{dataset_name}_FN.mat"), {
                'opt_out_scores': best_pure_fn_score.reshape(-1, 1),
                'label': labels.reshape(-1, 1)
            })

            # Iterate w
            for w in w_values:
                w = float(w) # pure float
                best_auc_for_w = -1
                best_score_for_w = None
                best_k_for_w = -1
                
                # Search best k for this w
                for k in k_values:
                    score = w * ksl_score + (1.0 - w) * fn_scores[k]
                    auc = roc_auc_score(labels, score)
                    if auc > best_auc_for_w:
                        best_auc_for_w = auc
                        best_score_for_w = score
                        best_k_for_w = k
                
                # Check global best
                if best_auc_for_w > best_overall_auc:
                    best_overall_auc = best_auc_for_w
                    best_overall_score = best_score_for_w
                    best_overall_w = w
                    best_overall_k = best_k_for_w
                
                # Format folder logic according to: "每一个w单独设置一个文件夹"
                w_folder_name = f"KSLFN_w_{w:.3f}_results"
                w_dir = os.path.join(results_base_dir, w_folder_name, dataset_name)
                os.makedirs(w_dir, exist_ok=True)
                
                # Exact file matching format: dataset_name_KSLFN_w_0.XXX.mat
                save_path = os.path.join(w_dir, f"{dataset_name}_KSLFN.mat")
                savemat(save_path, {
                    'opt_out_scores': best_score_for_w.reshape(-1, 1),
                    'label': labels.reshape(-1, 1),
                    'auc': best_auc_for_w,
                    'k': best_k_for_w,
                    'w': w
                })
                
                best_w_records.append({
                    'w': w,
                    'k': best_k_for_w,
                    'auc': best_auc_for_w
                })

            # Also store the very best config inside KSLFN_results properly
            kslfn_dir = os.path.join(results_base_dir, "KSLFN_results", dataset_name)
            os.makedirs(kslfn_dir, exist_ok=True)
            savemat(os.path.join(kslfn_dir, f"{dataset_name}_KSLFN.mat"), {
                'opt_out_scores': best_overall_score.reshape(-1, 1),
                'label': labels.reshape(-1, 1),
                'auc': best_overall_auc,
                'opt_k': best_overall_k,
                'opt_w': best_overall_w
            })
            
            # Log metrics
            all_metrics.append({
                'dataset': dataset_name,
                'best_overall_auc': best_overall_auc,
                'best_w': best_overall_w,
                'best_k': best_overall_k,
                'KSL_only_auc': best_pure_ksl_auc,
                'FN_only_auc': best_pure_fn_auc
            })
            print(f"  -> Best global AUC: {best_overall_auc:.4f} (w={best_overall_w:.3f}, k={best_overall_k})")
            
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {e}")

    # Save summary 
    df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(results_base_dir, 'all_outlier_result_summary.csv')
    df.to_csv(summary_path, index=False)
    print(f"\nSaved master summary to {summary_path}")
    print(f"All datasets processed. Anomaly scores generated inside Experimental_results folder!")

if __name__ == '__main__':
    main()
