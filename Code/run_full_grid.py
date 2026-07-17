"""
run_full_grid.py
================
对 4 个新数据集（mnist, pendigits, satimage2, mammography）进行全量参数网格搜索。
要求：
1. Min-Max 归一化 (X = (X-min)/(max-min))
2. w 范围 [0.025, 0.975]，步长 0.025 (排除 0 和 1)
3. k 范围 [10, 20, ..., 200] (覆盖之前 3D 图的范围)
4. 输出到 KSLFN_Remote/Experimental_results/grid_results/
"""
import os, sys
import numpy as np
import torch
from scipy.io import loadmat, savemat
from sklearn.metrics import roc_auc_score
from datetime import datetime

# 导入 KSLFN 核心逻辑
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

K_RANGE = list(range(10, 210, 10))  # 10, 20, ..., 200 (20个点)
W_RANGE = [round(x, 3) for x in np.arange(0.025, 1.0, 0.025)] # 39个点 (排除0, 1)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def min_max_norm(X):
    _min = X.min(axis=0)
    _max = X.max(axis=0)
    denom = _max - _min
    denom[denom == 0] = 1.0
    return (X - _min) / denom

# ─── 执行 ────────────────────────────────────────────────────────────────────
def run_grid():
    print(f"[{datetime.now()}] 开始全量网格实验 (Min-Max Normalization)")
    print(f"Device: {DEVICE}, K: {len(K_RANGE)}, W: {len(W_RANGE)}")

    for ds_name, ds_file in DATASETS.items():
        print(f"\nProcessing {ds_name}...")
        path = os.path.join(DATA_DIR, ds_file)
        if not os.path.exists(path):
            print(f"  Error: {path} not found"); continue
        
        # 加载并归一化
        try:
            mat = loadmat(path)
            X = mat['X'].astype(np.float32)
            y = mat['y'].ravel().astype(np.float32)
        except:
            print(f"  Error loading {ds_name}"); continue
            
        X_norm = min_max_norm(X)
        
        # 预计算 KSL 和 FN 基础得分 (避免重复计算)
        # 我们对每个 k 跑一次，得到 ksl_score 和 fn_score
        ds_out = os.path.join(OUTPUT_DIR, ds_name)
        os.makedirs(ds_out, exist_ok=True)
        
        grid_data = {} # (k, w) -> auc

        for k in K_RANGE:
            print(f"  k={k}...", end=" ", flush=True)
            # 计算 KSL 和 FN (w=0.5 只是为了占位，kslfn_fast 返回的是分解后的得分)
            # 注意：我们需要修改 KSLFN 或者直接手动调里面的核心逻辑来获取分解分
            # 这里调用一次，然后循环 w 组合
            try:
                # 为了效率，我们直接在这里做 fusion 循环
                # kslfn_fast 内部计算了相似度矩阵 R 和 FN
                # 我们假设 w 组合是在 CPU 上完成的（因为只是简单的加权）
                final_score, ksl_score, fn_score = kslfn_fast(X_norm, k, w=0.5, device=DEVICE)
                
                for w in W_RANGE:
                    combined = w * ksl_score + (1.0 - w) * fn_score
                    auc = roc_auc_score(y, combined.cpu().numpy())
                    grid_data[(k, w)] = auc
                print("done")
            except Exception as e:
                print(f"failed: {e}")

        # 保存为格式化数据以便 3D 绘图
        out_mat = os.path.join(ds_out, f"{ds_name}_grid.mat")
        # 转换成矩阵格式: rows=W, cols=K
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
        print(f"  Saved grid to {out_mat}")

if __name__ == "__main__":
    run_grid()
