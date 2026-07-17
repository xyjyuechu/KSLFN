"""
generate_plots.py — One-stop script for ROC curves, Nemenyi CD diagram, and Ablation study.
Outputs all figures to KSLFN_Remote/Figures/
Usage: python generate_plots.py
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
import matplotlib.gridspec as gridspec

try:
    import scikit_posthocs as sp
except ImportError:
    sp = None

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR           = '/Users/yijiachen/Documents/project/mine'
EXCEL_PATH         = f'{BASE_DIR}/Experimental_results/all_outlier_result-20250901.xls'
BASELINE_RESULTS   = f'{BASE_DIR}/Experimental_results'
KSLFN_RESULTS      = f'{BASE_DIR}/KSLFN_Remote/Experimental_results/KSLFN_results'
KSL_RESULTS        = f'{BASE_DIR}/KSLFN_Remote/Experimental_results/KSL_results'
FN_RESULTS         = f'{BASE_DIR}/KSLFN_Remote/Experimental_results/FN_results'
OUTPUT_DIR         = f'{BASE_DIR}/KSLFN_Remote/Figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/ROC', exist_ok=True)

# ─── Dataset mapping (display name → mat key) ─────────────────────────────────
DATASETS_28 = {
    'Abalone': 'abalone_variant1', 'Annthyroid': 'annthyroid',
    'Audiology': 'audiology_variant1', 'Breast': 'breast_cancer_variant1',
    'Chess_145': 'chess_nowin_145_variant1', 'Chess_185': 'chess_nowin_185_variant1',
    'Ecoli': 'ecoli', 'German': 'german_1_14_variant1', 'Glass': 'glass',
    'Ionosphere': 'ionosphere_b_24_variant1', 'Iris': 'iris_Irisvirginica_11_variant1',
    'Letter': 'letter', 'Lymphography': 'lymphography',
    'Monks_12': 'monks_0_12_variant1', 'Monks_4': 'monks_0_4_variant1',
    'Mushroom_365': 'mushroom_p_365_variant1', 'Mushroom_467': 'mushroom_p_467_variant1',
    'Sick_35': 'sick_sick_35_variant1', 'Sick_72': 'sick_sick_72_variant1',
    'Sonar': 'sonar_M_10_variant1', 'Spambase': 'spambase_spam_56_variant1',
    'Tic_26': 'tic_tac_toe_negative_26_variant1', 'Tic_32': 'tic_tac_toe_negative_32_variant1',
    'Vowels': 'vowels', 'Waveform': 'waveform_0_100_variant1', 'Wine': 'wine',
    'Yeast': 'yeast_ERL_5_variant1', 'Zoo': 'zoo_variant1',
}
IMAGE_DATASETS = {
    'MNIST': 'mnist', 'Pendigits': 'pendigits',
    'Satimage2': 'satimage2', 'Mammography': 'mammography',
}
ALL_DATASETS = dict(**DATASETS_28, **IMAGE_DATASETS)
ALL_DISPLAY_SORTED = sorted(ALL_DATASETS.keys(), key=lambda x: x.lower())

BASELINES_16 = ['SEQ','HBOS','IForest','ITB','WDOD','ODGrCR','NC','ApproE',
                 'COPOD','VarE','WNINOD','ECOD','ROD','FGAS','ILGNI','MFGAD']

# ROC: pick representative and visually distinct datasets (one from each domain)
ROC_DATASETS = ['Abalone', 'Glass', 'Ionosphere', 'Letter',
                 'Monks_12', 'Vowels', 'Zoo', 'MNIST',
                 'Pendigits', 'Satimage2', 'Mammography', 'Annthyroid']

# ─── Helpers ─────────────────────────────────────────────────────────────────
def load_kslfn(ds_key):
    p = f'{KSLFN_RESULTS}/{ds_key}/{ds_key}_KSLFN.mat'
    if not os.path.exists(p): return None, None
    d = loadmat(p)
    return d['opt_out_scores'].ravel(), d['label'].ravel()

def load_ksl(ds_key):
    p = f'{KSL_RESULTS}/{ds_key}/{ds_key}_KSL.mat'
    if not os.path.exists(p): return None
    d = loadmat(p)
    return d['opt_out_scores'].ravel()

def load_fn(ds_key):
    p = f'{FN_RESULTS}/{ds_key}/{ds_key}_FN.mat'
    if not os.path.exists(p): return None
    d = loadmat(p)
    return d['opt_out_scores'].ravel()

def load_baseline(method_name, ds_key):
    """Load baseline scores from Experimental_results folder."""
    method_dirs = [d for d in os.listdir(BASELINE_RESULTS)
                   if d.endswith('_results') and method_name.lower() in d.lower()
                   and os.path.isdir(os.path.join(BASELINE_RESULTS, d))]
    for mdir in method_dirs:
        ds_dir = os.path.join(BASELINE_RESULTS, mdir, ds_key)
        if not os.path.isdir(ds_dir): continue
        for f in os.listdir(ds_dir):
            if f.endswith('.mat') and f.startswith(ds_key):
                try:
                    d = loadmat(os.path.join(ds_dir, f))
                    scores = d['opt_out_scores']
                    return scores[:, 0]
                except: pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ROC CURVES
# ═══════════════════════════════════════════════════════════════════════════════
def generate_roc():
    print("\n--- Generating ROC curves ---")
    # We compare KSLFN vs representative baselines on each dataset
    roc_methods = ['KSLFN', 'HBOS', 'IForest', 'NC', 'FGAS', 'ECOD']
    colors     = ['#2ca02c','#3498DB','#E74C3C','#9B59B6','#F39C12','#1ABC9C']
    linestyles = ['-',       '--',     '-.',      ':',      '-',      '--']
    linewidths = [2.5,        1.5,      1.5,       1.5,      1.5,      1.5]

    for dname in ROC_DATASETS:
        ds_key = ALL_DATASETS.get(dname)
        if not ds_key: continue
        kslfn_scores, labels = load_kslfn(ds_key)
        if kslfn_scores is None: continue

        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        for method, color, ls, lw in zip(roc_methods, colors, linestyles, linewidths):
            if method == 'KSLFN':
                scores = kslfn_scores
            else:
                scores = load_baseline(method, ds_key)
            if scores is None or len(scores) != len(labels): continue
            try:
                fpr, tpr, _ = roc_curve(labels, scores)
                auc = roc_auc_score(labels, scores)
                ax.plot(fpr, tpr, label=f'{method} ({auc:.3f})',
                        color=color, linestyle=ls, linewidth=lw)
            except: pass

        ax.plot([0,1],[0,1],'k--',linewidth=0.8,alpha=0.5)
        ax.set_xlabel('FPR (%)', fontsize=12)
        ax.set_ylabel('TPR (%)', fontsize=12)
        ax.set_xticks([0,.2,.4,.6,.8,1])
        ax.set_xticklabels([0,20,40,60,80,100])
        ax.set_yticks([0,.2,.4,.6,.8,1])
        ax.set_yticklabels([0,20,40,60,80,100])
        ax.legend(fontsize=8, loc='lower right', ncol=1)
        ax.set_title(dname, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = f'{OUTPUT_DIR}/ROC/{dname}_ROC.pdf'
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f'  Saved {out_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NEMENYI CD DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════════
def generate_nemenyi():
    print("\n--- Generating Nemenyi CD diagram ---")
    if sp is None:
        print("  scikit_posthocs not installed. Run: pip install scikit-posthocs")
        return

    df = pd.read_excel(EXCEL_PATH)
    # Build AUC matrix for 28 original datasets
    records = []
    for dname, ds_key in DATASETS_28.items():
        row = df[df['dataset'] == ds_key]
        if len(row) == 0: continue
        r = {'dataset': dname}
        for b in BASELINES_16:
            r[b] = float(row[b].values[0]) if b in row.columns else np.nan
        # KSLFN
        ks, labs = load_kslfn(ds_key)
        r['KSLFN'] = roc_auc_score(labs, ks) if ks is not None else np.nan
        records.append(r)

    auc_df = pd.DataFrame(records).set_index('dataset')
    all_methods = BASELINES_16 + ['KSLFN']
    auc_df = auc_df[all_methods].dropna()

    k = len(all_methods)
    n = len(auc_df)
    # Compute average ranks (lower rank = better)
    ranks_arr = np.apply_along_axis(lambda row: rankdata(-row, method='average'), 1, auc_df.values)
    avg_ranks = pd.Series(ranks_arr.mean(axis=0), index=all_methods)

    # CD
    q_alpha = studentized_range.isf(0.05, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6. * n))

    # Plot
    palette = {'KSLFN': '#2ca02c'}
    ordered = avg_ranks.sort_values()
    split = (k + 1) // 2
    left_labels = ordered.index[:split].tolist()
    right_labels = ordered.index[split:].tolist()

    spacing = 0.8
    start_y = -1.1
    max_rows = max(len(left_labels), len(right_labels))
    lower_bound = start_y - spacing * max_rows - 0.4

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(lower_bound, 1.6)
    ax.xaxis.set_ticks_position('top')
    tick_pos = np.arange(1, k+1, dtype=float)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(v) for v in range(k, 0, -1)], fontsize=10)
    ax.yaxis.set_visible(False)
    for spine in ('left','right','bottom'): ax.spines[spine].set_visible(False)
    ax.spines['top'].set_position('zero')
    ax.spines['top'].set_linewidth(1.6)
    for t in tick_pos: ax.vlines(t, -0.25, 0.25, color='black', linewidth=1.2)

    left_margin, right_margin = 0.6, k + 0.4
    top_color, default_color = '#0055b3', '#1a1a1a'

    for idx, label in enumerate(left_labels):
        y = start_y - idx*spacing
        color = palette.get(label, top_color)
        x = float(avg_ranks[label])
        ax.plot([x,x],[0,y],color=color,linewidth=1.2)
        ax.plot([left_margin,x],[y,y],color=color,linewidth=1.2)
        ax.text(left_margin-0.05,y,label,ha='right',va='center',fontsize=9,color=color,fontweight='bold' if label=='KSLFN' else 'normal')

    for idx, label in enumerate(right_labels):
        y = start_y - idx*spacing
        color = palette.get(label, default_color)
        x = float(avg_ranks[label])
        ax.plot([x,x],[0,y],color=color,linewidth=1.2)
        ax.plot([x,right_margin],[y,y],color=color,linewidth=1.2)
        ax.text(right_margin+0.05,y,label,ha='left',va='center',fontsize=9,color=color,fontweight='bold' if label=='KSLFN' else 'normal')

    # Crossbar
    best = [ordered.index[0]]
    for lab, val in ordered.iloc[1:].items():
        if val - ordered.iloc[0] <= cd + 1e-9: best.append(lab)
        else: break
    crossbar_h = None
    if len(best) > 1:
        xs = sorted(float(avg_ranks[b]) for b in best)
        crossbar_h = 1.1
        ax.hlines(crossbar_h, xs[0], xs[-1], color='#cc1f1f', linewidth=2.2)

    center = (tick_pos[0] + tick_pos[-1]) / 2
    cd_h = (crossbar_h + 0.25) if crossbar_h else 0.9
    ax.hlines(cd_h, center - cd/2, center + cd/2, color='#cc1f1f', linewidth=2.2)
    ax.vlines(center-cd/2, cd_h-0.07, cd_h+0.07, color='#cc1f1f', linewidth=2.2)
    ax.vlines(center+cd/2, cd_h-0.07, cd_h+0.07, color='#cc1f1f', linewidth=2.2)
    ax.text((center-cd/2+center+cd/2)/2, cd_h+0.1, f'CD={cd:.4f}',
            color='#cc1f1f', ha='center', va='bottom', fontsize=11)

    ax.set_title('Nemenyi test on AUC (28 datasets, α=0.05)', fontsize=13, pad=18)
    plt.tight_layout()
    for ext in ['pdf','png']:
        plt.savefig(f'{OUTPUT_DIR}/nemenyi_cd.{ext}', bbox_inches='tight', dpi=200)
    plt.close()
    print(f'  Saved {OUTPUT_DIR}/nemenyi_cd.pdf/png')
    print(f'  CD={cd:.4f}, KSLFN avg rank={avg_ranks["KSLFN"]:.4f}')
    print(f'  Rank 1 method: {ordered.index[0]} (rank={ordered.iloc[0]:.4f})')


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════════════
def generate_ablation():
    print("\n--- Generating Ablation study plot ---")
    results = {}
    print_rows = []
    for dname, ds_key in sorted(ALL_DATASETS.items(), key=lambda x: x[0].lower()):
        ks_sc, labs = load_kslfn(ds_key)
        ksl_sc = load_ksl(ds_key)
        fn_sc  = load_fn(ds_key)
        if ks_sc is None: continue
        try:
            kslfn_auc = roc_auc_score(labs, ks_sc)
            ksl_auc   = roc_auc_score(labs, ksl_sc) if ksl_sc is not None else np.nan
            fn_auc    = roc_auc_score(labs, fn_sc) if fn_sc is not None else np.nan
            results[dname] = {'KSLFN': kslfn_auc, 'KSL': ksl_auc, 'FN': fn_auc}
            print_rows.append(f'  {dname:20s}  KSL={ksl_auc:.4f}  FN={fn_auc:.4f}  KSLFN={kslfn_auc:.4f}')
        except Exception as e:
            print(f'  Warning {dname}: {e}')

    for row in print_rows: print(row)

    ds_names = list(results.keys())
    kslfn_vals = [results[d]['KSLFN'] for d in ds_names]
    ksl_vals   = [results[d]['KSL']   for d in ds_names]
    fn_vals    = [results[d]['FN']    for d in ds_names]

    x = np.arange(len(ds_names))
    w = 0.28
    fig, ax = plt.subplots(figsize=(max(14, len(ds_names)*0.55), 5.5))
    ax.bar(x - w, ksl_vals,   w, label='KSL only',   color='#3498DB', alpha=0.85, edgecolor='white')
    ax.bar(x,     fn_vals,    w, label='FN only',    color='#E74C3C', alpha=0.85, edgecolor='white')
    ax.bar(x + w, kslfn_vals, w, label='KSLFN',   color='#2ca02c', alpha=0.85, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, ncol=3, loc='lower right')
    ax.set_title('Ablation Study: KSL-only vs FN-only vs KSLFN', fontsize=13, fontweight='bold')

    # Add averages textbox
    avg_ksl   = np.nanmean(ksl_vals)
    avg_fn    = np.nanmean(fn_vals)
    avg_kslfn = np.nanmean(kslfn_vals)
    textstr = f'Avg: KSL={avg_ksl:.4f}  FN={avg_fn:.4f}  KSLFN={avg_kslfn:.4f}'
    ax.text(0.01, 0.99, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    for ext in ['pdf','png']:
        plt.savefig(f'{OUTPUT_DIR}/ablation_study.{ext}', bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved {OUTPUT_DIR}/ablation_study.pdf/png')


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    generate_roc()
    generate_nemenyi()
    generate_ablation()
    print(f'\nAll figures saved to {OUTPUT_DIR}')
