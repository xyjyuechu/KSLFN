"""
run_full_grid_normalized_scores.py
==================================
对 4 个新数据集跑全量参数网格 (k x w)。
关键点：对 KSL 和 FN 的 raw scores 分别进行 Min-Max 归一化后再融合。
1. w 范围 [0.025, 0.975]，步长 0.025 (排除 0 和 1)
2. k 范围 [10, 20, ..., 200]
3. 归一化：score = (score - min) / (max - min)
"""
import os, sys
import numpy as np
import torch
from scipy.io import loadmat, savemat
from sklearn.metrics import roc_auc_score
from datetime import datetime

sys.path.append('/Users/yijiachen/Documents/project/mine/KSLFN_Remote/Code')
from KSLFN import kslfn_fast

# ─── 配置 ────────────────────────────────────────────────────────────────────
DATA_DIR   = '/Users/yijiachen/Documents/project/mine/Code_to_DrY/Tran_data_of_outlier_detection/Numerical'
OUTPUT_DIR = '/Users/yijiachen/Documents/project/mine/KSLFN_Remote/Experimental_results/grid_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = {
    'mnist': 'mnist.mat',
    'pendigits': 'pendigits.mat',
    'satimage2': 'satimage2.mat',
    'mammography': 'mammography.mat'
}

K_RANGE = list(range(10, 210, 10))
W_RANGE = [round(x, 3) for x in np.arange(0.025, 1.0, 0.025)]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def min_max_normalize_scores(s):
    # s is a tensor
    s_min = s.min()
    s_max = s.max()
    denom = s_max - s_min
    if denom == 0:
        return torch.zeros_like(s)
    return (s - s_min) / denom

# ─── 执行 ────────────────────────────────────────────────────────────────────
def run_grid():
    print(f"[{datetime.now()}] 开始全量网格实验 (Min-Max Normalized Scores)")
    
    for ds_name, ds_file in DATASETS.items():
        print(f"\nProcessing {ds_name}...")
        path = os.path.join(DATA_DIR, ds_file)
        if not os.path.exists(path): continue
        
        mat = loadmat(path)
        X = mat['X'].astype(np.float32)
        y = mat['y'].ravel().astype(np.float32)
        
        # 特征归一化 (保持原来的 max 归一化或常规处理，核心是得分的归一化)
        X_norm = X / (X.max(axis=0) + 1e-9)
        
        grid_data = {}

        for k in K_RANGE:
            try:
                # 获取原始 KSL 和 FN 得分
                _, ksl_score, fn_score = kslfn_fast(X_norm, k, w=0.5, device=DEVICE)
                
                # 对两种得分分别进行 Min-Max 归一化
                ksl_norm = min_max_normalize_scores(ksl_score)
                fn_norm  = min_max_normalize_scores(fn_score)
                
                # 循环 w 生成网格
                ksl_np = ksl_norm.cpu().numpy()
                fn_np  = fn_norm.cpu().numpy()
                
                for w in W_RANGE:
                    combined = w * ksl_np + (1.0 - w) * fn_np
                    auc = roc_auc_score(y, combined)
                    grid_data[(k, w)] = auc
                sys.stdout.write('.')
                sys.stdout.flush()
            except Exception as e:
                print(f"\nk={k} failed: {e}")

        # 保存为 .mat 给 3D 绘图使用
        out_mat = os.path.join(OUTPUT_DIR, f"{ds_name}_grid.mat")
        auc_matrix = np.zeros((len(W_RANGE), len(K_RANGE)))
        for i, w in enumerate(W_RANGE):
            for j, k in enumerate(K_RANGE):
                auc_matrix[i, j] = grid_data.get((k, w), 0.0)
        
        savemat(out_mat, {
            'auc_matrix': auc_matrix,
            'k_values': np.array(K_RANGE),
            'w_values': np.array(W_RANGE),
            'dataset': ds_name
        })
        print(f"\n  Saved grid for {ds_name}")

if __name__ == "__main__":
    run_grid()
