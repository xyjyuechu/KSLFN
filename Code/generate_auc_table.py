"""
Generate LaTeX AUC comparison tables matching the paper format.
Reads baseline data from Experimental_results Excel file and KSLFN results from post_process output.

Usage: python generate_auc_table.py
"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR     = '/Users/yijiachen/Documents/project/mine'
EXCEL_PATH   = f'{BASE_DIR}/Experimental_results/all_outlier_result-20250901.xls'
KSLFN_RESULTS_DIR = f'{BASE_DIR}/KSLFN_Remote/Experimental_results/KSLFN_results'

# Our 28 original datasets — display name → mat file name
DATASETS_28 = {
    'Abalone':      'abalone_variant1',
    'Annthyroid':   'annthyroid',
    'Audiology':    'audiology_variant1',
    'Breast':       'breast_cancer_variant1',
    'Chess\_145':   'chess_nowin_145_variant1',
    'Chess\_185':   'chess_nowin_185_variant1',
    'Ecoli':        'ecoli',
    'German':       'german_1_14_variant1',
    'Glass':        'glass',
    'Ionosphere':   'ionosphere_b_24_variant1',
    'Iris':         'iris_Irisvirginica_11_variant1',
    'Letter':       'letter',
    'Lymphography': 'lymphography',
    'Monks\_12':    'monks_0_12_variant1',
    'Monks\_4':     'monks_0_4_variant1',
    'Mushroom\_365':'mushroom_p_365_variant1',
    'Mushroom\_467':'mushroom_p_467_variant1',
    'Sick\_35':     'sick_sick_35_variant1',
    'Sick\_72':     'sick_sick_72_variant1',
    'Sonar':        'sonar_M_10_variant1',
    'Spambase':     'spambase_spam_56_variant1',
    'Tic\_26':      'tic_tac_toe_negative_26_variant1',
    'Tic\_32':      'tic_tac_toe_negative_32_variant1',
    'Vowels':       'vowels',
    'Waveform':     'waveform_0_100_variant1',
    'Wine':         'wine',
    'Yeast':        'yeast_ERL_5_variant1',
    'Zoo':          'zoo_variant1',
}

# 4 image datasets — display name → mat basename (no folder suffix)  
IMAGE_DATASETS = {
    'MNIST':       'mnist',
    'Pendigits':   'pendigits',
    'Satimage2':   'satimage2',
    'Mammography': 'mammography',   # update when result ready
}

# Baselines in the Excel (column names)
BASELINES_P1 = ['SEQ', 'HBOS', 'IForest', 'ITB', 'WDOD', 'ODGrCR', 'NC', 'ApproE']
BASELINES_P2 = ['COPOD', 'VarE', 'WNINOD', 'ECOD', 'ROD', 'FGAS', 'ILGNI', 'MFGAD']
# ─────────────────────────────────────────────────────────────────────────────

def get_kslfn_auc(ds_key):
    """Get KSLFN AUC for a dataset from saved results."""
    mat_path = os.path.join(KSLFN_RESULTS_DIR, ds_key, f'{ds_key}_KSLFN.mat')
    if not os.path.exists(mat_path):
        return np.nan
    try:
        data = loadmat(mat_path)
        scores = data['opt_out_scores'].ravel()
        labels = data['label'].ravel()
        return roc_auc_score(labels, scores)
    except Exception as e:
        print(f'  Warning: {ds_key}: {e}')
        return np.nan

def fmt(val, bold=False, ref=None):
    """Format AUC value, bold if best in row, add arrow vs ref."""
    if np.isnan(val):
        return '--'
    s = f'{val:.4f}'
    if bold:
        s = f'\\textbf{{{s}}}'
    if ref is not None:
        diff = val - ref
        if diff > 0.0001:
            s += f'\\textcolor{{red}}{{\\tiny~↑{abs(diff):.4f}}}'
        elif diff < -0.0001:
            s += f'\\textcolor{{green}}{{\\tiny~↓{abs(diff):.4f}}}'
    return s

def make_table(display_names, ds_keys, df_excel, baselines, part_num):
    """Build a LaTeX tabular body."""
    kslfn_aucs = {k: get_kslfn_auc(v) for k, v in zip(display_names, ds_keys)}
    rows = []
    kslfn_col = []
    baseline_cols = {b: [] for b in baselines}

    for dname, dkey in zip(display_names, ds_keys):
        row_excel = df_excel[df_excel['dataset'] == dkey]
        kslfn = kslfn_aucs[dname]
        kslfn_col.append(kslfn)
        
        bl_vals = {}
        for b in baselines:
            if len(row_excel) > 0 and b in row_excel.columns:
                bl_vals[b] = float(row_excel[b].values[0])
            else:
                bl_vals[b] = np.nan
            baseline_cols[b].append(bl_vals[b])

        # Find max per row (including KSLFN)
        all_vals = [kslfn] + [bl_vals[b] for b in baselines]
        max_val = max(v for v in all_vals if not np.isnan(v))

        # Compute best competitor (for arrow reference)
        best_competitor = max((bl_vals[b] for b in baselines if not np.isnan(bl_vals[b])), default=np.nan)

        cells = [dname]
        kslfn_str = fmt(kslfn, bold=(abs(kslfn - max_val) < 1e-4 if not np.isnan(kslfn) else False),
                       ref=best_competitor)
        cells.append(kslfn_str)
        for b in baselines:
            v = bl_vals[b]
            cells.append(fmt(v, bold=(not np.isnan(v) and abs(v - max_val) < 1e-4)))
        rows.append(' & '.join(cells) + ' \\\\')

    # Average row
    avg_kslfn = np.nanmean(kslfn_col)
    avg_best_comp = max(np.nanmean(baseline_cols[b]) for b in baselines)
    avg_cells = ['Average']
    avg_cells.append(fmt(avg_kslfn, bold=True, ref=avg_best_comp))
    for b in baselines:
        avg_v = np.nanmean(baseline_cols[b])
        avg_cells.append(fmt(avg_v))
    rows.append('\\midrule')
    rows.append(' & '.join(avg_cells) + ' \\\\')

    return rows

def generate_tables():
    df = pd.read_excel(EXCEL_PATH)
    
    # All datasets: 28 original + 4 image, sorted alphabetically by display name
    all_combined = dict(**DATASETS_28, **IMAGE_DATASETS)
    all_display = sorted(all_combined.keys(), key=lambda x: x.replace('\\', '').lower())
    all_keys = [all_combined[d] for d in all_display]

    for part_num, baselines in [(1, BASELINES_P1), (2, BASELINES_P2)]:
        rows = make_table(all_display, all_keys, df, baselines, part_num)
        
        col_spec = '@{} l c ' + 'c' * len(baselines) + ' @{}'
        bl_header = ' & '.join(baselines)
        
        table = f"""\\begin{{table}}[!htb]
\\centering
\\caption{{AUC Comparison on All 32 Datasets (Part {part_num})}}
\\label{{tab:auc_all_part{part_num}}}
\\setlength{{\\tabcolsep}}{{3pt}}
\\renewcommand{{\\arraystretch}}{{1.1}}
\\footnotesize
\\begin{{tabular}}{{{col_spec}}}
\\toprule
Dataset & KSLFN (Ours) & {bl_header} \\\\
\\midrule
"""
        for r in rows:
            table += r + '\n'
        table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        out_path = f'/tmp/auc_table_part{part_num}.tex'
        with open(out_path, 'w') as f:
            f.write(table)
        print(f'Part {part_num} saved to {out_path}')
        print(table[:300])
        print('...')

if __name__ == '__main__':
    generate_tables()
