"""
generate_final_plots.py
=======================
从 KSLFN_Remote/Experimental_results 读取新实验结果，生成：
1. 4 个新数据集的 3D 参数分析图 (= KSLFN_IJAR/Figures/*.pdf)  ← 完全参照 redraw_param_sensitivity.py
2. 4 个新数据集的 ROC 曲线  (= KSLFN_IJAR/Figures/*_ROC.pdf)  ← 完全参照 plot/plot.py
3. 涵盖 32 个数据集的 Nemenyi CD 图 (覆盖旧 nemenyi_cd_auc.pdf) ← 完全参照 nemenyi_cd_plot.py

Usage: python generate_final_plots.py
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import rankdata, studentized_range
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import scikit_posthocs as sp

# 把 analysis/ 加入路径以复用 nemenyi_cd_plot.py 的内部函数
sys.path.insert(0, '/Users/yijiachen/Documents/project/mine/analysis')
from nemenyi_cd_plot import _compute_cd, _build_palette, _spread_positions, _plot_cd_diagram

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = '/Users/yijiachen/Documents/project/mine'
REMOTE_EXP      = f'{BASE_DIR}/KSLFN_Remote/Experimental_results'   # ← 新实验结果在这里
OLD_EXP         = f'{BASE_DIR}/Experimental_results'                  # ← 原始 baseline 结果
FIGURES_DIR     = f'{BASE_DIR}/KSLFN_IJAR/Figures'
EXCEL_PATH      = f'{OLD_EXP}/all_outlier_result-20250901.xls'

KSLFN_RESULTS   = f'{REMOTE_EXP}/KSLFN_results'
KSL_RESULTS     = f'{REMOTE_EXP}/KSL_results'
FN_RESULTS      = f'{REMOTE_EXP}/FN_results'

# ─── Dataset info ─────────────────────────────────────────────────────────────
DATASETS_28 = {
    'abalone_variant1': 'Abalone', 'annthyroid': 'Annthyroid',
    'audiology_variant1': 'Audiology', 'breast_cancer_variant1': 'Breast',
    'chess_nowin_145_variant1': 'Chess_145', 'chess_nowin_185_variant1': 'Chess_185',
    'ecoli': 'Ecoli', 'german_1_14_variant1': 'German', 'glass': 'Glass',
    'ionosphere_b_24_variant1': 'Ionosphere', 'iris_Irisvirginica_11_variant1': 'Iris',
    'letter': 'Letter', 'lymphography': 'Lymphography',
    'monks_0_12_variant1': 'Monks_12', 'monks_0_4_variant1': 'Monks_4',
    'mushroom_p_365_variant1': 'Mushroom_365', 'mushroom_p_467_variant1': 'Mushroom_467',
    'sick_sick_35_variant1': 'Sick_35', 'sick_sick_72_variant1': 'Sick_72',
    'sonar_M_10_variant1': 'Sonar', 'spambase_spam_56_variant1': 'Spambase',
    'tic_tac_toe_negative_26_variant1': 'Tic_26', 'tic_tac_toe_negative_32_variant1': 'Tic_32',
    'vowels': 'Vowels', 'waveform_0_100_variant1': 'Waveform',
    'wine': 'Wine', 'yeast_ERL_5_variant1': 'Yeast', 'zoo_variant1': 'Zoo',
}
IMAGE_DATASETS = {
    'mnist': 'MNIST', 'pendigits': 'Pendigits',
    'satimage2': 'Satimage2', 'mammography': 'Mammography',
}
ALL_DATASETS = dict(**DATASETS_28, **IMAGE_DATASETS)
BASELINES_16 = ['SEQ','HBOS','IForest','ITB','WDOD','ODGrCR','NC','ApproE',
                 'COPOD','VarE','WNINOD','ECOD','ROD','FGAS','ILGNI','MFGAD']

# 与原 plot.py 完全一致
COLORS  = ["#3498DB","#E74C3C","#2ECC71","#F1C40F","#9B59B6",
           "#34495E","#1ABC9C","#E67E22","#95A5A6","#27AE60",
           "#C0392B","#16A085","#28B463","#E84393","#F39C12","#E84393"]
MARKERS = ["o","v","^","<",">","8","s","p","*","h","H","D","d","P","X"]


# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_kslfn(ds_key):
    p = f'{KSLFN_RESULTS}/{ds_key}/{ds_key}_KSLFN.mat'
    if not os.path.exists(p): return None, None
    d = loadmat(p)
    return d['opt_out_scores'].ravel(), d['label'].ravel()

def find_baseline_file(method_name, ds_key):
    for d in os.listdir(OLD_EXP):
        if not d.endswith('_results') or not os.path.isdir(os.path.join(OLD_EXP, d)): continue
        if method_name.lower() not in d.lower(): continue
        ds_dir = os.path.join(OLD_EXP, d, ds_key)
        if not os.path.isdir(ds_dir): continue
        for f in sorted(os.listdir(ds_dir)):
            if f.endswith('.mat') and f.startswith(ds_key):
                return os.path.join(ds_dir, f)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 3D 参数分析图 for 4 new datasets — 完全参照 redraw_param_sensitivity.py
# ═══════════════════════════════════════════════════════════════════════════════
def generate_param_surfaces():
    """从 KSLFN_w_*/dataset/*.mat 读数据构建 k×w AUC 矩阵，再画 3D surface"""
    print("\n--- 3D Parameter Sensitivity plots (4 new datasets) ---")

    # 收集所有 w 目录（排除 w=0 和 w=1）
    w_dirs = sorted([
        d for d in os.listdir(REMOTE_EXP)
        if d.startswith('KSLFN_w_') and d.endswith('_results')
        and os.path.isdir(os.path.join(REMOTE_EXP, d))
    ])

    for ds_key in IMAGE_DATASETS.keys():
        # 收集 k 和 w 维度上的 AUC
        k2w2auc = {}  # k_idx -> w -> best_auc

        for w_dir in w_dirs:
            w_str = w_dir.replace('KSLFN_w_', '').replace('_results', '')
            try: w_val = float(w_str)
            except: continue
            if w_val <= 0.0 or w_val >= 1.0: continue  # 排除 0 和 1

            ds_dir = os.path.join(REMOTE_EXP, w_dir, ds_key)
            if not os.path.isdir(ds_dir): continue
            for f in os.listdir(ds_dir):
                if not f.endswith('.mat'): continue
                try:
                    d = loadmat(os.path.join(ds_dir, f))
                    auc = float(d.get('auc', [0.0]).ravel()[0])
                    k   = int(d.get('k', [-1]).ravel()[0])
                    if k < 0: continue
                    k2w2auc.setdefault(k, {})[w_val] = auc
                except: continue

        if not k2w2auc:
            print(f'  Skip {ds_key}: no data'); continue

        all_k = sorted(k2w2auc.keys())
        all_w = sorted(next(iter(k2w2auc.values())).keys())
        auc_matrix = np.array([[k2w2auc[k].get(w, np.nan) for w in all_w] for k in all_k])
        # shape: (len_k, len_w), for 3D: rows=w, cols=k
        auc_matrix_T = auc_matrix.T  # (len_w, len_k)

        k_arr = np.array(all_k, dtype=float)
        w_arr = np.array(all_w, dtype=float)

        # ─── 完全参照 redraw_param_sensitivity.py 的画法 ─────────────────────
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.fontset'] = 'stix'

        k_idx = np.arange(len(k_arr))
        w_idx = np.arange(len(w_arr))
        K, W = np.meshgrid(k_idx, w_idx)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(K, W, auc_matrix_T, cmap='viridis',
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
        print(f'  ✅ 已保存参数分析图: {ds_key}')


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ROC for 4 new datasets — 完全参照 plot/plot.py
# ═══════════════════════════════════════════════════════════════════════════════
def generate_roc_new_datasets():
    print("\n--- ROC curves (4 new datasets) ---")
    plot_methods = ['KSLFN', 'HBOS', 'IForest', 'COPOD', 'FGAS', 'ECOD', 'NC', 'WNINOD']

    for ds_key in IMAGE_DATASETS.keys():
        kslfn_scores, labels = load_kslfn(ds_key)
        if kslfn_scores is None:
            print(f'  Skip {ds_key}: no KSLFN results'); continue

        cur_fpr, cur_tpr, cur_methods = [], [], []

        # KSLFN
        try:
            fpr, tpr, _ = roc_curve(labels, kslfn_scores)
            cur_fpr.append(fpr); cur_tpr.append(tpr); cur_methods.append('KSLFN')
        except: pass

        # Baselines
        for method in plot_methods[1:]:
            f = find_baseline_file(method, ds_key)
            if f is None: continue
            try:
                res = loadmat(f)
                scores = res.get('opt_out_scores', res.get('opt_out_score'))[:, 0]
                fpr, tpr, _ = roc_curve(labels, scores)
                cur_fpr.append(fpr); cur_tpr.append(tpr); cur_methods.append(method)
            except: pass

        # ─── 完全参照 plot.py 格式 ────────────────────────────────────────────
        fig = plt.figure(dpi=150)
        for j in range(len(cur_methods)):
            color = '#E74C3C' if cur_methods[j] == 'KSLFN' else COLORS[j % len(COLORS)]
            marker = '*' if cur_methods[j] == 'KSLFN' else MARKERS[j % len(MARKERS)]
            lw = 2.0 if cur_methods[j] == 'KSLFN' else 1.0
            plt.plot(cur_fpr[j], cur_tpr[j], label=cur_methods[j],
                     color=color, marker=marker, markevery=3, linewidth=lw)

        plt.xticks(ticks=[0,0.2,0.4,0.6,0.8,1], labels=[0,20,40,60,80,100])
        plt.yticks(ticks=[0,0.2,0.4,0.6,0.8,1], labels=[0,20,40,60,80,100])
        plt.xlabel(r"FPR", fontdict={"size": 20})
        plt.ylabel(r"TPR", fontdict={"size": 20})
        plt.xlim(-0.05, None)
        plt.ylim(-0.05, None)
        plt.legend(ncol=3, loc="lower center", prop={"size": 13})

        out_path = f'{FIGURES_DIR}/{ds_key}_ROC.pdf'
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f'  ✅ 已保存 ROC: {ds_key}_ROC.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Nemenyi 32 datasets — 完全参照 nemenyi_cd_plot.py
# ═══════════════════════════════════════════════════════════════════════════════
def generate_nemenyi_32():
    print("\n--- Nemenyi CD diagram (32 datasets) ---")
    df = pd.read_excel(EXCEL_PATH)

    records = []
    for ds_key in ALL_DATASETS:
        row = df[df['dataset'] == ds_key]
        ks, labs = load_kslfn(ds_key)
        if ks is None: continue
        r = {'dataset': ds_key, 'KSLFN': roc_auc_score(labs, ks)}
        for b in BASELINES_16:
            if len(row) > 0 and b in row.columns:
                v = row[b].values[0]
                r[b] = float(v) if not np.isnan(float(v)) else np.nan
            else:
                r[b] = np.nan
        records.append(r)

    auc_df = pd.DataFrame(records).set_index('dataset')
    all_methods = BASELINES_16 + ['KSLFN']
    auc_df = auc_df[all_methods].dropna()
    print(f'  Using {len(auc_df)} complete datasets')

    k  = len(all_methods)
    n  = len(auc_df)
    ranks_arr = np.apply_along_axis(
        lambda row: rankdata(-row, method='average'), 1, auc_df.values)
    avg_ranks = pd.Series(ranks_arr.mean(axis=0), index=all_methods)
    nemenyi   = sp.posthoc_nemenyi_friedman(-auc_df)
    cd_value  = _compute_cd(k, n)

    palette = _build_palette()
    fig = _plot_cd_diagram(avg_ranks, nemenyi, palette, cd_value)

    # 覆盖原有 nemenyi_cd_auc.pdf
    out_pdf = f'{FIGURES_DIR}/nemenyi_cd_auc.pdf'
    out_png = f'{FIGURES_DIR}/nemenyi_cd_auc.png'
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f'  ✅ 已保存 {out_pdf}')
    print(f'  CD={cd_value:.4f}, KSLFN avg rank={avg_ranks["KSLFN"]:.4f}')


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    generate_param_surfaces()
    generate_roc_new_datasets()
    generate_nemenyi_32()
    print(f'\n全部完成！图片在 {FIGURES_DIR}')
