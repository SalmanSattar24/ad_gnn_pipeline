#!/usr/bin/env python
"""
Test runner for Step 1B: Cell-Type Deconvolution
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

RAW_DATA_DIR = "../data/raw"
PROCESSED_DATA_DIR = "../data/processed"
RESULTS_DIR = "../results/step1"

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CELL_TYPES = ['Ex', 'In', 'Ast', 'Oli', 'Mic', 'OPCs']

print("\n" + "="*70)
print("STEP 1B: TEST - CELL-TYPE DECONVOLUTION")
print("="*70 + "\n")

# ============================================================================
# GENERATE SYNTHETIC MATHYS REFERENCE
# ============================================================================
print("[SETUP] Generating synthetic Mathys 2019 reference...")

np.random.seed(42)

n_cells = 8066
n_genes = 10000
n_subjects = 48
cell_types = ['Ex', 'In', 'Ast', 'Oli', 'Mic', 'OPCs']
cell_type_proportions = [0.35, 0.15, 0.20, 0.15, 0.10, 0.05]

cell_type_labels = np.random.choice(cell_types, size=n_cells, p=cell_type_proportions)
subject_ids = np.random.choice([f'ROSMAP_{i:03d}' for i in range(n_subjects)], size=n_cells)
gene_names = [f'GENE_{i}' for i in range(n_genes)]

X_counts = np.zeros((n_cells, n_genes))
for ct in cell_types:
    mask = cell_type_labels == ct
    n_ct_cells = mask.sum()
    ct_baseline = np.random.exponential(scale=0.5, size=n_genes)
    ct_noise = np.random.normal(0, 1, size=(n_ct_cells, n_genes))
    X_counts[mask, :] = np.maximum(ct_baseline[np.newaxis, :] + ct_noise, 0)

adata = sc.AnnData(
    X=X_counts,
    obs=pd.DataFrame({
        'cell_type': cell_type_labels,
        'broad.cell.type': cell_type_labels,
        'subject_id': subject_ids
    }, index=[f'cell_{i}' for i in range(n_cells)]),
    var=pd.DataFrame({'gene_id': gene_names}, index=gene_names)
)

adata.write(f'{RAW_DATA_DIR}/mathys_reference.h5ad')
print(f"  Saved: {RAW_DATA_DIR}/mathys_reference.h5ad")
print(f"  Shape: {adata.shape}")
print(f"  Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")

# ============================================================================
# GENERATE SYNTHETIC BULK PROTEOMICS
# ============================================================================
print("\n[SETUP] Generating synthetic bulk proteomics...")

n_patients = 180
n_proteins = 1500

protein_indices = np.random.choice(n_genes, size=n_proteins, replace=False)
selected_genes = [gene_names[i] for i in protein_indices]

reference_means = np.zeros((len(cell_types), n_genes))
for i, ct in enumerate(cell_types):
    ct_mask = adata.obs['cell_type'] == ct
    reference_means[i, :] = adata.X[ct_mask, :].mean(axis=0)

bulk_data = np.zeros((n_patients, n_proteins))
ground_truth_proportions = []

for p in range(n_patients):
    proportions = np.random.dirichlet([1.0] * len(cell_types))
    ground_truth_proportions.append(proportions)
    bulk_profile = proportions @ reference_means[np.ix_(np.arange(len(cell_types)), protein_indices)]
    noise = np.random.normal(0, 0.1, size=n_proteins)
    bulk_data[p, :] = np.maximum(bulk_profile + noise, 0)

bulk_df = pd.DataFrame(
    bulk_data,
    index=[f'patient_{i:03d}' for i in range(n_patients)],
    columns=selected_genes
)
bulk_df.index.name = 'patient_id'
bulk_df.to_csv(f'{PROCESSED_DATA_DIR}/rosmap_proteomics_cleaned.csv')
print(f"  Saved: {PROCESSED_DATA_DIR}/rosmap_proteomics_cleaned.csv")
print(f"  Shape: {bulk_df.shape}")

diagnoses = np.random.choice(['Control', 'AD'], size=n_patients, p=[0.55, 0.45])
metadata_df = pd.DataFrame({
    'diagnosis': diagnoses,
    'age_death': np.random.randint(60, 100, n_patients),
    'msex': np.random.choice([0, 1], n_patients),
    'pmi': np.random.uniform(2, 30, n_patients)
}, index=bulk_df.index)
metadata_df.index.name = 'projid'
metadata_df.to_csv(f'{PROCESSED_DATA_DIR}/rosmap_metadata.csv')
print(f"  Saved: {PROCESSED_DATA_DIR}/rosmap_metadata.csv")
print(f"  Diagnosis: Control={sum(diagnoses=='Control')}, AD={sum(diagnoses=='AD')}")

# ============================================================================
# DECONVOLUTION FUNCTIONS
# ============================================================================

def load_mathys_reference(file_path):
    print("\n[1] Loading Mathys 2019 snRNA-seq reference...")
    adata = sc.read_h5ad(file_path)
    print(f"  Loaded .h5ad: {adata.shape}")
    print(f"  Variables (genes): {adata.shape[1]}")
    print(f"  Observations (cells): {adata.shape[0]}")
    return adata

def load_bulk_proteomics(file_path):
    print("[2] Loading bulk proteomics data...")
    df = pd.read_csv(file_path, index_col=0)
    print(f"  Shape: {df.shape[0]} samples x {df.shape[1]} proteins")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    return df

def load_metadata(file_path):
    print("[3] Loading clinical metadata...")
    df = pd.read_csv(file_path, index_col=0)
    print(f"  Shape: {df.shape}")
    print(f"  Diagnosis counts: {df['diagnosis'].value_counts().to_dict()}")
    return df

def extract_cell_type_reference(adata, cell_type_col='cell_type', cell_types=None):
    print("[4] Building cell-type reference profiles...")
    if cell_types is None:
        cell_types = adata.obs[cell_type_col].unique()
    gene_names = np.array(adata.var.index)
    n_genes = len(gene_names)
    reference_profiles = np.zeros((len(cell_types), n_genes))
    for i, ct in enumerate(cell_types):
        mask = adata.obs[cell_type_col] == ct
        n_cells = mask.sum()
        if hasattr(adata.X, 'toarray'):
            reference_profiles[i, :] = adata.X[mask, :].toarray().mean(axis=0)
        else:
            reference_profiles[i, :] = adata.X[mask, :].mean(axis=0)
        print(f"  {ct}: {n_cells} cells, mean expression computed")
    return reference_profiles, cell_types, gene_names

def match_genes_to_proteins(reference_genes, bulk_proteins):
    print("[5] Matching reference genes to bulk proteins...")
    reference_genes_set = set(reference_genes)
    bulk_proteins_set = set(bulk_proteins)
    overlapping = reference_genes_set.intersection(bulk_proteins_set)
    print(f"  Reference genes: {len(reference_genes)}")
    print(f"  Bulk proteins: {len(bulk_proteins)}")
    print(f"  Overlapping features: {len(overlapping)} ({len(overlapping)/len(bulk_proteins)*100:.1f}%)")
    ref_indices = np.array([i for i, g in enumerate(reference_genes) if g in overlapping])
    bulk_indices = np.array([i for i, p in enumerate(bulk_proteins) if p in overlapping])
    return ref_indices, bulk_indices, sorted(list(overlapping))

def nnls_deconvolve(bulk_row, reference_profiles):
    def objective(x):
        residuals = bulk_row - reference_profiles.T @ x
        return np.sum(residuals ** 2)
    def sum_constraint(x):
        return np.sum(x) - 1.0
    x0 = np.ones(reference_profiles.shape[0]) / reference_profiles.shape[0]
    bounds = [(0, 1) for _ in range(reference_profiles.shape[0])]
    constraints = {'type': 'eq', 'fun': sum_constraint}
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    proportions = result.x / result.x.sum()
    return proportions

def run_deconvolution(bulk_df, reference_profiles, cell_types):
    print("[6] Running NNLS deconvolution...")
    n_samples = bulk_df.shape[0]
    n_proteins = bulk_df.shape[1]
    n_cell_types = reference_profiles.shape[0]
    cell_type_proportions = np.zeros((n_samples, n_cell_types))
    deconvolved_profiles = np.zeros((n_samples, n_proteins, n_cell_types))
    for i, (sample_id, bulk_row) in enumerate(bulk_df.iterrows()):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{n_samples} samples...")
        proportions = nnls_deconvolve(bulk_row.values, reference_profiles)
        cell_type_proportions[i, :] = proportions
        for j in range(n_cell_types):
            deconvolved_profiles[i, :, j] = reference_profiles[j, :] * proportions[j]
    print(f"  Completed deconvolution for {n_samples} samples")
    return cell_type_proportions, deconvolved_profiles

def save_deconvolution_results(bulk_df, cell_type_proportions, deconvolved_profiles, cell_types, processed_dir):
    print("[7] Saving deconvolution results...")
    proportions_df = pd.DataFrame(cell_type_proportions, index=bulk_df.index, columns=cell_types)
    proportions_file = f"{processed_dir}/cell_type_proportions.csv"
    proportions_df.to_csv(proportions_file)
    print(f"  Saved: {proportions_file}")
    print(f"  Shape: {proportions_df.shape}")

    deconvolved_list = []
    for i, sample_id in enumerate(bulk_df.index):
        for j, protein_id in enumerate(bulk_df.columns):
            for k, ct in enumerate(cell_types):
                deconvolved_list.append({
                    'sample_id': sample_id,
                    'protein_id': protein_id,
                    'cell_type': ct,
                    'abundance': deconvolved_profiles[i, j, k]
                })
    deconvolved_df = pd.DataFrame(deconvolved_list)
    deconvolved_file = f"{processed_dir}/deconvolved_profiles.csv"
    deconvolved_df.to_csv(deconvolved_file, index=False)
    print(f"  Saved: {deconvolved_file}")
    print(f"  Shape: {deconvolved_df.shape}")
    return proportions_df, deconvolved_df

def plot_cell_type_proportions(proportions_df, metadata_df, results_dir):
    print("[8] Generating cell-type proportions visualization...")
    data = proportions_df.copy()
    data['diagnosis'] = metadata_df.loc[data.index, 'diagnosis'].values
    data = data.sort_values(by=['diagnosis', data.columns[0]])

    fig, ax = plt.subplots(figsize=(16, 6))
    cell_types = [c for c in data.columns if c != 'diagnosis']
    colors = plt.cm.Set3(np.linspace(0, 1, len(cell_types)))
    bottom = np.zeros(len(data))
    for i, ct in enumerate(cell_types):
        ax.bar(range(len(data)), data[ct], bottom=bottom, label=ct, color=colors[i], alpha=0.8)
        bottom += data[ct].values

    diagnosis_change = data['diagnosis'].ne(data['diagnosis'].shift()).cumsum()
    for pos in np.where(diagnosis_change.diff().fillna(0) != 0)[0]:
        ax.axvline(x=pos - 0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Patients (sorted by diagnosis)', fontsize=12)
    ax.set_ylabel('Cell-Type Proportion', fontsize=12)
    ax.set_title('Estimated Cell-Type Proportions per Patient', fontsize=13, fontweight='bold')
    ax.legend(title='Cell Type', loc='upper left', fontsize=9)
    ax.set_ylim([0, 1])
    plt.xticks([])
    fig_file = f"{results_dir}/cell_type_proportions.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_file}")
    plt.close()

def plot_celltype_comparison(proportions_df, metadata_df, results_dir):
    print("[9] Generating cell-type comparison visualization...")
    data = proportions_df.copy()
    data['diagnosis'] = metadata_df.loc[data.index, 'diagnosis'].values
    cell_types = proportions_df.columns.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, ct in enumerate(cell_types):
        ax = axes[idx]
        plot_data = []
        for diagnosis in ['Control', 'AD']:
            values = data[data['diagnosis'] == diagnosis][ct].values
            plot_data.append(values)
        parts = ax.violinplot(plot_data, positions=[0, 1], showmeans=True, showmedians=True)
        stat, pval = mannwhitneyu(plot_data[0], plot_data[1])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Control', 'AD'])
        ax.set_ylabel('Proportion', fontsize=11)
        ax.set_title(f'{ct}\n(p={pval:.2e})', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([data[ct].min() - 0.05, data[ct].max() + 0.05])

    plt.tight_layout()
    fig_file = f"{results_dir}/celltype_proportion_comparison.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_file}")
    plt.close()

def test_significant_changes(proportions_df, metadata_df):
    print("\n[10] Statistical testing of cell-type differences...")
    print("="*70)
    data = proportions_df.copy()
    data['diagnosis'] = metadata_df.loc[data.index, 'diagnosis'].values
    control_data = data[data['diagnosis'] == 'Control']
    ad_data = data[data['diagnosis'] == 'AD']
    print(f"\nControl samples: {len(control_data)}")
    print(f"AD samples: {len(ad_data)}")
    print("\nCell-Type Proportion Differences (Mann-Whitney U test):")
    print("-" * 70)

    significant_changes = {}
    for ct in proportions_df.columns:
        control_vals = control_data[ct].values
        ad_vals = ad_data[ct].values
        stat, pval = mannwhitneyu(control_vals, ad_vals)
        control_mean = control_vals.mean()
        ad_mean = ad_vals.mean()
        direction = 'increased' if ad_mean > control_mean else 'decreased'
        fold_change = ad_mean / control_mean if control_mean > 0 else np.inf
        sig = "*" if pval < 0.05 else ""
        print(f"{ct:8s}: Control={control_mean:.3f}, AD={ad_mean:.3f} (p={pval:.4f}) {direction} {sig}")
        if pval < 0.05:
            significant_changes[ct] = {'pval': pval, 'direction': direction, 'fold_change': fold_change}

    print("\n" + "="*70)
    print(f"\nSignificant changes (p<0.05):")
    if significant_changes:
        for ct, info in significant_changes.items():
            print(f"  - {ct}: {info['direction']} (fold change: {info['fold_change']:.2f}, p={info['pval']:.4f})")
    else:
        print("  None")
    return significant_changes

# ============================================================================
# MAIN PIPELINE
# ============================================================================
try:
    adata_ref = load_mathys_reference(f"{RAW_DATA_DIR}/mathys_reference.h5ad")
    bulk_df = load_bulk_proteomics(f"{PROCESSED_DATA_DIR}/rosmap_proteomics_cleaned.csv")
    metadata_df = load_metadata(f"{PROCESSED_DATA_DIR}/rosmap_metadata.csv")

    ref_profiles, cell_types_list, ref_genes = extract_cell_type_reference(
        adata_ref, cell_type_col='cell_type', cell_types=CELL_TYPES
    )

    ref_idx, bulk_idx, overlapping_features = match_genes_to_proteins(ref_genes, bulk_df.columns.values)

    ref_profiles_matched = ref_profiles[:, ref_idx]
    bulk_df_matched = bulk_df.iloc[:, bulk_idx]
    print(f"\n  Using {len(overlapping_features)} overlapping features for deconvolution")

    cell_type_proportions, deconvolved_profiles = run_deconvolution(bulk_df_matched, ref_profiles_matched, CELL_TYPES)

    proportions_df, deconvolved_df = save_deconvolution_results(
        bulk_df_matched, cell_type_proportions, deconvolved_profiles, CELL_TYPES, PROCESSED_DATA_DIR
    )

    plot_cell_type_proportions(proportions_df, metadata_df, RESULTS_DIR)
    plot_celltype_comparison(proportions_df, metadata_df, RESULTS_DIR)

    significant_changes = test_significant_changes(proportions_df, metadata_df)

    print("\n" + "="*70)
    print("STEP 1B DECONVOLUTION COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {PROCESSED_DATA_DIR}")
    print(f"Figures saved to: {RESULTS_DIR}")
    print("Ready for Step 1C: Pseudotime & NMF Analysis\n")

except Exception as e:
    print(f"\nERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
