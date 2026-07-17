"""
generate_final_plots_v2.py
==========================
Step 1: 参数分析图 (4 new datasets) — 从 KSLFN_Remote/Experimental_results 读.mat，
        完全按照 redraw_param_sensitivity.py 的 3D surface 样式绘图
Step 2: ROC 曲线 (4 new datasets) — 按照 plot/plot.py 的图格式
Step 3: Nemenyi (32 datasets) — 生成扩展 CSV，调用 nemenyi_cd_plot.py 生成

Usage: python generate_final_plots_v2.py
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = '/Users/yijiachen/Documents/project/mine'
REMOTE_EXP    = f'{BASE_DIR}/KSLFN_Remote/Experimental_results'
OLD_EXP       = f'{BASE_DIR}/Experimental_results'
FIGURES_DIR   = f'{BASE_DIR}/KSLFN_IJAR/Figures'
ANALYSIS_OUT  = f'{BASE_DIR}/analysis/output'
EXCEL_PATH    = f'{OLD_EXP}/all_outlier_result-20250901.xls'
ORIG_CSV      = f'{ANALYSIS_OUT}/kslfn_baseline16_sota20.csv'   # 原28数据集CSV
NEW_CSV       = f'{ANALYSIS_OUT}/kslfn_baseline16_sota32.csv'   # 新32数据集CSV

KSLFN_RESULTS = f'{REMOTE_EXP}/KSLFN_results'

IMAGE_DATASETS = ['mnist', 'pendigits', 'satimage2', 'mammography']
BASELINES_16 = ['SEQ','HBOS','IForest','ITB','WDOD','ODGrCR','NC','ApproE',
                 'COPOD','VarE','WNINOD','ECOD','ROD','FGAS','ILGNI','MFGAD']

# 与 plot/plot.py 完全一致
COLORS  = ["#3498DB","#E74C3C","#2ECC71","#F1C40F","#9B59B6",
           "#34495E","#1ABC9C","#E67E22","#95A5A6","#27AE60",
           "#C0392B","#16A085","#28B463","#E84393","#F39C12","#E84393"]
MARKERS = ["o","v","^","<",">","8","s","p","*","h","H","D","d","P","X"]


def load_kslfn(ds_key):
    p = f'{KSLFN_RESULTS}/{ds_key}/{ds_key}_KSLFN.mat'
    if not os.path.exists(p): return None, None
    d = loadmat(p)
    return d['opt_out_scores'].ravel(), d['label'].ravel()

def find_baseline_file(method_name, ds_key):
    for d in sorted(os.listdir(OLD_EXP)):
        if not d.endswith('_results') or not os.path.isdir(os.path.join(OLD_EXP, d)): continue
        if method_name.lower() not in d.lower(): continue
        ds_dir = os.path.join(OLD_EXP, d, ds_key)
        if not os.path.isdir(ds_dir): continue
        for f in os.listdir(ds_dir):
            if f.endswith('.mat') and f.startswith(ds_key):
                return os.path.join(ds_dir, f)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: 3D 参数分析图 — 完全照 redraw_param_sensitivity.py
# ═══════════════════════════════════════════════════════════════════════════════
def generate_param_surfaces():
    """
    从 KSLFN_Remote/Experimental_results/KSLFN_w_xxx_results/ds/ds.mat
    读取每个 (w, k) 组合的 AUC，拼成 auc_matrix，再按原脚本画 3D surface
    """
    print("\n=== STEP 1: 3D 参数分析图 ===")

    # 收集所有 w 目录（0 < w < 1）
    w_dirs_sorted = sorted([
        d for d in os.listdir(REMOTE_EXP)
        if d.startswith('KSLFN_w_') and d.endswith('_results')
        and os.path.isdir(os.path.join(REMOTE_EXP, d))
    ], key=lambda x: float(x.replace('KSLFN_w_', '').replace('_results', '')))

    for ds_key in IMAGE_DATASETS:
        # { w_val -> { k_val -> auc } }
        data = {}
        for w_dir in w_dirs_sorted:
            w_str = w_dir.replace('KSLFN_w_', '').replace('_results', '')
            try: w_val = float(w_str)
            except: continue
            if w_val <= 0.0 or w_val >= 1.0: continue  # 排除 0 和 1

            ds_dir = os.path.join(REMOTE_EXP, w_dir, ds_key)
            if not os.path.isdir(ds_dir): continue
            for f in os.listdir(ds_dir):
                if not f.endswith('.mat'): continue
                try:
                    m = loadmat(os.path.join(ds_dir, f))
                    auc_v = float(m['auc'].ravel()[0])
                    k_v   = int(m['k'].ravel()[0])
                    data.setdefault(w_val, {})[k_v] = auc_v
                except: continue

        if not data:
            print(f'  ⚠️ {ds_key}: 没有找到数据，跳过'); continue

        all_w = sorted(data.keys())
        all_k = sorted(set(k for kd in data.values() for k in kd.keys()))

        # auc_matrix shape: (len_w, len_k)  — 跟 redraw 脚本一致 (W=行, K=列)
        auc_matrix = np.array([
            [data[w].get(k, np.nan) for k in all_k]
            for w in all_w
        ])

        k_arr = np.array(all_k, dtype=float)
        w_arr = np.array(all_w, dtype=float)

        # 逻辑坐标轴（避免刻度密集）
        k_idx = np.arange(len(k_arr))
        w_idx = np.arange(len(w_arr))
        K, W = np.meshgrid(k_idx, w_idx)

        # ─── 完全照 redraw_param_sensitivity.py ───
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(K, W, auc_matrix, cmap='viridis',
                               edgecolor='black', alpha=0.8, linewidth=0.1,
                               rstride=4, cstride=4)

        ax.set_xlabel(r'$k$', fontsize=20, labelpad=8)
        ax.set_ylabel(r'$w$', fontsize=20, labelpad=8)
        ax.set_zlabel('AUC', fontsize=20, labelpad=14)

        step_k = max(1, len(k_arr) // 5)
        ax.set_xticks(k_idx[::step_k])
        ax.set_xticklabels([f"{int(k_arr[i])}" for i in range(0, len(k_arr), step_k)], fontsize=14)

        step_w = max(1, len(w_arr) // 6)
        ax.set_yticks(w_idx[::step_w])
        ax.set_yticklabels([f"{w_arr[i]:.2f}".rstrip('0').rstrip('.')
                            for i in range(0, len(w_arr), step_w)], fontsize=14)

        ax.tick_params(axis='z', labelsize=16)
        ax.view_init(elev=30, azim=45)
        ax.set_box_aspect([1.2, 1.2, 1])

        cbar = fig.colorbar(surf, ax=ax, pad=0.02, shrink=0.6, aspect=20)
        cbar.ax.tick_params(labelsize=16)
        plt.subplots_adjust(left=0.10, right=0.98, bottom=0.03, top=0.99)

        out_pdf = f'{FIGURES_DIR}/{ds_key}.pdf'
        out_png = f'{FIGURES_DIR}/{ds_key}.png'
        plt.savefig(out_pdf, format='pdf', bbox_inches=None, dpi=300, pad_inches=1.0)
        plt.savefig(out_png, format='png', bbox_inches=None, dpi=300, pad_inches=1.0)
        plt.close(fig)
        print(f'  ✅ {ds_key}.pdf  (k={len(all_k)} values, w={len(all_w)} values)')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: ROC 曲线 — 全部 32 个数据集，格式完全照 plot 3/plot.py
# ═══════════════════════════════════════════════════════════════════════════════

# 16 个 baseline 方法名 → Experimental_results 中对应目录名
BASELINE_DIRS = {
    'SEQ':    'SEQ_results_FCM',
    'HBOS':   'HBOS_results',
    'IForest':'IForest_results',
    'ITB':    'ITB_results_FCM',
    'WDOD':   'WDOD_results_FCM',
    'ODGrCR': 'ODGrCR_results_FCM',
    'NC':     'NC_results',
    'ApproE': 'ApproE_results',
    'COPOD':  'COPOD_results',
    'VarE':   'VarE_results',
    'WNINOD': 'WNINOD_results',
    'ECOD':   'ECOD_results',
    'ROD':    'ROD_results',
    'FGAS':   'FGAS_results',
    'ILGNI':  'ILGNI_results',
    'MFGAD':  'MFGAD_results',
}

def _find_ds_dir(method_dir, ds_key):
    """大小写不敏感查找数据集子目录"""
    if not os.path.isdir(method_dir): return None
    ds_low = ds_key.lower()
    for name in os.listdir(method_dir):
        if name.lower() == ds_low and os.path.isdir(os.path.join(method_dir, name)):
            return os.path.join(method_dir, name)
    return None

def _find_mat(ds_dir, ds_key):
    """大小写不敏感查找主 .mat（排除 _lam- _sigma- _k- _lambda- _thelta- 等参数变体）"""
    ds_low = ds_key.lower()
    variant_suffixes = ('_lam-', '_sigma-', '_k-', '_lambda-', '_thelta-')
    for f in os.listdir(ds_dir):
        fl = f.lower()
        if fl.startswith(ds_low) and f.endswith('.mat') and not any(s in fl for s in variant_suffixes):
            return os.path.join(ds_dir, f)
    return None

def _load_scores(mat_path):
    """只用 opt_out_scores / opt_out_score（主文件的标准 key）"""
    res = loadmat(mat_path)
    for key in ('opt_out_scores', 'opt_out_score'):
        if key in res:
            sc = res[key]
            return sc[:, 0] if sc.ndim > 1 else sc.ravel()
    return None

def generate_roc_new():
    """
    为全部 32 个数据集生成 ROC 图。
    格式完全照 plot 3/plot.py：先画 16 个 baseline，再把 KSLFN 压在最上层（黑色虚线）。
    缺失方法直接跳过，不填对角线 fallback。
    """
    print("\n=== STEP 2: ROC 曲线（32 datasets）===")
    ROC_OUT_DIR = f'{BASE_DIR}/KSLFN_Remote/Figures/ROC'
    os.makedirs(ROC_OUT_DIR, exist_ok=True)

    datasets_32 = sorted([
        f.replace('.mat', '')
        for f in os.listdir(f'{BASE_DIR}/KSLFN_Remote/Datasets')
        if f.endswith('.mat')
    ])

    for ds_key in datasets_32:
        kslfn_scores, labels = load_kslfn(ds_key)
        if kslfn_scores is None:
            print(f'  ⚠️ {ds_key}: 无 KSLFN 结果，跳过'); continue

        base_fpr, base_tpr, base_names = [], [], []
        for method_name, mdir in BASELINE_DIRS.items():
            ds_dir = _find_ds_dir(os.path.join(OLD_EXP, mdir), ds_key)
            if ds_dir is None: continue
            f_path = _find_mat(ds_dir, ds_key)
            if f_path is None: continue
            try:
                sc = _load_scores(f_path)
                if sc is None or len(sc) != len(labels): continue
                fpr, tpr, _ = roc_curve(labels, sc)
                base_fpr.append(fpr); base_tpr.append(tpr)
                base_names.append(method_name)
            except: continue

        fig = plt.figure(dpi=150)
        for j, name in enumerate(base_names):
            plt.plot(base_fpr[j], base_tpr[j], label=name,
                     color=COLORS[j % len(COLORS)],
                     marker=MARKERS[j % len(MARKERS)],
                     markevery=3, linewidth=1.0)
        try:
            fpr_k, tpr_k, _ = roc_curve(labels, kslfn_scores)
            plt.plot(fpr_k, tpr_k, label='KSLFN',
                     color='black', linestyle='--', linewidth=2.0)
        except: pass

        plt.xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
                   labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
                   labels=['0%', '20%', '40%', '60%', '80%', '100%'])
        plt.xlabel(r"FPR", fontdict={"size": 20})
        plt.ylabel(r"TPR", fontdict={"size": 20})
        plt.xlim(-0.05, None)
        plt.ylim(-0.05, None)
        plt.legend(ncol=3, loc="lower center", prop={"size": 13})
        nice = ds_key[0].upper() + ds_key[1:]
        plt.savefig(f'{ROC_OUT_DIR}/{nice}_ROC.pdf', bbox_inches='tight')
        plt.savefig(f'{FIGURES_DIR}/{ds_key}_ROC.pdf', bbox_inches='tight')
        plt.close()
        print(f'  ✅ {ds_key}_ROC.pdf  (baselines={len(base_names)})')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: 生成包含 32 个数据集的 CSV，调用原版 nemenyi_cd_plot.py
# ═══════════════════════════════════════════════════════════════════════════════
def generate_nemenyi_csv_and_plot():
    print("\n=== STEP 3: Nemenyi (32 datasets) ===")

    # 读原 28 数据集 CSV
    orig = pd.read_csv(ORIG_CSV)
    print(f'  原 CSV: {len(orig)} 行, 列: {orig.columns.tolist()}')

    # 读 Excel baseline
    df_excel = pd.read_excel(EXCEL_PATH)

    new_rows = []
    for ds_key in IMAGE_DATASETS:
        row_excel = df_excel[df_excel['dataset'] == ds_key]
        ks, labs  = load_kslfn(ds_key)
        if ks is None:
            print(f'  ⚠️ skip {ds_key}'); continue
        kslfn_auc = roc_auc_score(labs, ks)

        r = {'dataset': ds_key}
        for b in BASELINES_16:
            if len(row_excel) > 0 and b in row_excel.columns:
                v = row_excel[b].values[0]
                r[b] = float(v) if not np.isnan(float(v)) else np.nan
            else:
                r[b] = np.nan
        r['KSLFN'] = kslfn_auc
        new_rows.append(r)
        print(f'  Added {ds_key}: KSLFN={kslfn_auc:.4f}')

    # 合并并保存
    combined = pd.concat([orig, pd.DataFrame(new_rows)], ignore_index=True)
    combined.to_csv(NEW_CSV, index=False)
    print(f'  新 CSV 共 {len(combined)} 行，已保存到 {NEW_CSV}')

    # 调用原版 nemenyi_cd_plot.py，修改其读取路径
    print('\n  调用原版 nemenyi_cd_plot.py ...')
    sys.path.insert(0, f'{BASE_DIR}/analysis')
    import importlib
    import nemenyi_cd_plot as ncd
    # patch BASELINE_PATH to point to new CSV
    import pathlib
    ncd.BASELINE_PATH = pathlib.Path(NEW_CSV)
    ncd.ABLATION_PATH = pathlib.Path(f'{ANALYSIS_OUT}/kslfn_ablation_28.csv')
    ncd.OUT_DIR       = pathlib.Path(ANALYSIS_OUT)
    ncd.main()

    # 覆盖到 KSLFN_IJAR/Figures/
    import shutil
    src = f'{ANALYSIS_OUT}/nemenyi_cd_auc.pdf'
    dst = f'{FIGURES_DIR}/nemenyi_cd_auc.pdf'
    shutil.copy2(src, dst)
    print(f'  ✅ Nemenyi 图已复制到 {dst}')


if __name__ == '__main__':
    generate_param_surfaces()
    generate_roc_new()
    generate_nemenyi_csv_and_plot()
    print(f'\n✅ 全部完成！图片在 {FIGURES_DIR}')
