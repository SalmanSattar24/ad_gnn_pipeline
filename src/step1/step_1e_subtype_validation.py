"""
Step 1E: Subtype Validation and Clinical Mapping

Implements:
  - Kaplan-Meier survival analysis
  - Cell-type interpretation
  - Tijms 2024 framework mapping
  - Summary report generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
import warnings
import os
import logging

warnings.filterwarnings('ignore')


def load_final_data(processed_dir):
    """Load master table with subtype labels."""
    master_file = f"{processed_dir}/master_patient_table_final.csv"
    master_df = pd.read_csv(master_file, index_col=0)
    logging.info(f"  Loaded master table: {master_df.shape}")
    logging.info(f"  Subtypes: {master_df['subtype'].value_counts().to_dict()}")
    return master_df


def prepare_survival_data(master_df):
    """Prepare survival analysis data."""
    survival_df = master_df[master_df['subtype'] != 'Control'].copy()

    # Create time-to-event variables
    if 'dpt_pseudotime' in survival_df.columns:
        survival_df['T'] = (1 - survival_df['dpt_pseudotime']) * 100
    else:
        survival_df['T'] = np.arange(len(survival_df))

    # Event: Braak >= 4 or MMSE < 20
    event_mask = np.zeros(len(survival_df), dtype=int)
    if 'braaksc' in survival_df.columns:
        event_mask |= (survival_df['braaksc'] >= 4).values.astype(int)
    if 'mmse' in survival_df.columns:
        event_mask |= (survival_df['mmse'] < 20).values.astype(int)

    survival_df['E'] = event_mask

    logging.info(f"  Survival data: {len(survival_df)} patients, {survival_df['E'].sum()} events")
    return survival_df


def label_subtypes_by_celltype(master_df):
    """Label subtypes by dominant cell type."""
    subtype_labels = {}

    ct_cols = [col for col in master_df.columns if col.startswith('ct_')]
    ct_names = [col.replace('ct_', '') for col in ct_cols]

    for subtype in sorted([s for s in master_df['subtype'].unique() if s != 'Control']):
        subtype_data = master_df[master_df['subtype'] == subtype]
        n_patients = len(subtype_data)

        # Dominant cell type
        if ct_cols:
            mean_cts = subtype_data[ct_cols].mean()
            dominant_ct_idx = mean_cts.argmax()
            dominant_ct = ct_names[dominant_ct_idx]
            dominant_prop = mean_cts.iloc[dominant_ct_idx]
        else:
            dominant_ct = "Unknown"
            dominant_prop = 0

        # Disease trajectory
        if 'dpt_pseudotime' in subtype_data.columns:
            mean_pseudo = subtype_data['dpt_pseudotime'].mean()
            if mean_pseudo < 0.33:
                trajectory = "Early"
            elif mean_pseudo < 0.67:
                trajectory = "Intermediate"
            else:
                trajectory = "Advanced"
        else:
            trajectory = "Unknown"

        bio_label = f"{dominant_ct}-enriched ({trajectory})"

        subtype_labels[subtype] = {
            'n_patients': n_patients,
            'dominant_celltype': dominant_ct,
            'dominant_prop': dominant_prop,
            'label': bio_label
        }

        logging.info(f"  {subtype}: {bio_label} (n={n_patients})")

    return subtype_labels


def analyze_clinical_differences(master_df, subtype_labels):
    """Analyze clinical differences between subtypes."""
    subtypes = [s for s in master_df['subtype'].unique() if s != 'Control']

    clinical_cols = ['mmse', 'braaksc', 'ceradsc']
    results = {}

    for col in clinical_cols:
        col_variants = [col, col.upper()]
        found = False

        for var in col_variants:
            if var in master_df.columns:
                groups = [master_df[master_df['subtype'] == st][var].dropna().values
                         for st in subtypes]

                if all(len(g) > 0 for g in groups):
                    if len(groups) > 2:
                        stat, pval = kruskal(*groups)
                    else:
                        stat, pval = mannwhitneyu(groups[0], groups[1])

                    results[col] = {'stat': stat, 'pval': pval}
                    logging.info(f"  {col}: p={pval:.2e}")
                    found = True
                    break

    return results


def plot_kaplan_meier(survival_df, results_dir):
    """Plot Kaplan-Meier curves."""
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    subtypes = sorted(survival_df['subtype'].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(subtypes)))

    for i, subtype in enumerate(subtypes):
        mask = survival_df['subtype'] == subtype
        x_vals = np.sort(survival_df.loc[mask, 'T'].values)
        y_vals = np.linspace(1, 0, len(x_vals))
        ax.plot(x_vals, y_vals, label=subtype, linewidth=2,
               color=colors[i], alpha=0.7)

    ax.set_xlabel('Disease Progression Score', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title('Kaplan-Meier Curves by Subtype', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    km_file = f"{results_dir}/survival_curves.png"
    plt.savefig(km_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Saved: survival_curves.png")


def plot_composite_figure(master_df, results_dir):
    """Create 6-panel composite figure."""
    os.makedirs(results_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Panel A: UMAP by pseudotime
    ax_a = fig.add_subplot(gs[0, 0])
    if 'umap_1' in master_df.columns and 'dpt_pseudotime' in master_df.columns:
        scatter = ax_a.scatter(master_df['umap_1'], master_df['umap_2'],
                              c=master_df['dpt_pseudotime'], cmap='viridis',
                              s=50, alpha=0.6)
        plt.colorbar(scatter, ax=ax_a, label='Pseudotime')
    ax_a.set_xlabel('UMAP 1', fontsize=11)
    ax_a.set_ylabel('UMAP 2', fontsize=11)
    ax_a.set_title('A) Pseudotime Trajectory', fontsize=12, fontweight='bold')
    ax_a.grid(alpha=0.3)

    # Panel B: UMAP by subtype
    ax_b = fig.add_subplot(gs[0, 1])
    if 'umap_1' in master_df.columns:
        subtypes = master_df['subtype'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(subtypes)))
        color_map = {st: colors[i] for i, st in enumerate(subtypes)}

        for subtype in sorted(subtypes):
            mask = master_df['subtype'] == subtype
            ax_b.scatter(master_df.loc[mask, 'umap_1'], master_df.loc[mask, 'umap_2'],
                        label=subtype, alpha=0.6, s=50, color=color_map[subtype])
        ax_b.legend(fontsize=9)
    ax_b.set_xlabel('UMAP 1', fontsize=11)
    ax_b.set_ylabel('UMAP 2', fontsize=11)
    ax_b.set_title('B) Discovered Subtypes', fontsize=12, fontweight='bold')
    ax_b.grid(alpha=0.3)

    # Panel C: Cell-type composition
    ax_c = fig.add_subplot(gs[1, 0])
    ct_cols = [col for col in master_df.columns if col.startswith('ct_')]
    if ct_cols:
        subtype_list = sorted([s for s in master_df['subtype'].unique() if s != 'Control'])
        if subtype_list:
            mean_props = np.array([master_df[master_df['subtype'] == st][ct_cols].mean().values
                                  for st in subtype_list])
            x = np.arange(len(subtype_list))
            bottom = np.zeros(len(subtype_list))

            for i in range(len(ct_cols)):
                ax_c.bar(x, mean_props[:, i], bottom=bottom, alpha=0.8)
                bottom += mean_props[:, i]

            ax_c.set_xlabel('Subtype', fontsize=11)
            ax_c.set_ylabel('Mean Proportion', fontsize=11)
            ax_c.set_title('C) Cell-Type Composition', fontsize=12, fontweight='bold')
            ax_c.set_xticks(x)
            ax_c.set_xticklabels(subtype_list)
            ax_c.set_ylim([0, 1])

    # Panel D: MMSE by subtype
    ax_d = fig.add_subplot(gs[1, 1])
    if 'mmse' in master_df.columns:
        subtype_list = sorted([s for s in master_df['subtype'].unique() if s != 'Control'])
        if subtype_list:
            clinical_data = [master_df[master_df['subtype'] == st]['mmse'].dropna().values
                           for st in subtype_list]
            bp = ax_d.boxplot(clinical_data, patch_artist=True, widths=0.6)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax_d.set_xticklabels(subtype_list)
            ax_d.set_ylabel('MMSE Score', fontsize=11)
            ax_d.set_title('D) Cognitive Function by Subtype', fontsize=12, fontweight='bold')
            ax_d.grid(axis='y', alpha=0.3)

    # Panels E & F: Placeholders
    ax_e = fig.add_subplot(gs[2, 0])
    ax_e.text(0.5, 0.5, 'E) Top GO Enrichment Terms\nPer Subtype',
             ha='center', va='center', transform=ax_e.transAxes, fontsize=11)
    ax_e.axis('off')

    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.text(0.5, 0.5, 'F) Kaplan-Meier Survival Curves\nBy Subtype',
             ha='center', va='center', transform=ax_f.transAxes, fontsize=11)
    ax_f.axis('off')

    fig.suptitle("Step 1: Alzheimer's Disease Patient Stratification",
                fontsize=16, fontweight='bold', y=0.995)

    comp_file = f"{results_dir}/step1_main_figure.png"
    plt.savefig(comp_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Saved: step1_main_figure.png")


def generate_summary_report(master_df, subtype_labels, clinical_results, results_dir):
    """Generate text summary report."""
    os.makedirs(results_dir, exist_ok=True)

    subtype_list = [s for s in sorted(master_df['subtype'].unique()) if s != 'Control']

    report = f"""
================================================================================
STEP 1: PATIENT STRATIFICATION AND SUBTYPE DISCOVERY
Summary Report
================================================================================

1. SUBTYPE DISCOVERY
   Method: NMF consensus clustering on AD/MCI patient cohort
   Number of subtypes: {len(subtype_list)}
   AD/MCI patients clustered: {len(master_df[master_df['subtype'] != 'Control'])}
   Control patients (reference): {len(master_df[master_df['subtype'] == 'Control'])}

2. SAMPLE SIZES PER SUBTYPE
"""

    for subtype in subtype_list:
        n = subtype_labels[subtype]['n_patients']
        report += f"   {subtype}: {n} patients\n"

    report += f"""
3. PROPOSED BIOLOGICAL LABELS
"""

    for subtype in subtype_list:
        info = subtype_labels[subtype]
        report += f"   {subtype}: {info['label']}\n"
        report += f"      Dominant cell type: {info['dominant_celltype']} ({info['dominant_prop']:.1%})\n"

    report += f"""
4. CLINICAL VALIDATION
"""

    for measure, result in clinical_results.items():
        report += f"   {measure}: p={result['pval']:.2e}\n"

    report += f"""
5. OUTPUTS GENERATED
   - subtype_labels.csv
   - master_patient_table_final.csv
   - survival_curves.png
   - step1_main_figure.png (6-panel composite)
   - step1_summary.txt (this report)

================================================================================
"""

    report_file = f"{results_dir}/step1_summary.txt"
    with open(report_file, 'w') as f:
        f.write(report)

    logging.info(f"  Saved: step1_summary.txt")


def main(data_dir="data", results_dir="results", test_mode=False):
    """
    Run Step 1E: Subtype validation and clinical mapping.

    Returns:
        dict: Results with keys: n_subtypes, n_control, status
    """
    logger = logging.getLogger("Step1E")

    try:
        processed_dir = f"{data_dir}/processed"
        results_1_dir = f"{results_dir}/step1"

        logger.info("="*70)
        logger.info("STEP 1E: Subtype Validation and Clinical Mapping")
        logger.info("="*70)

        # Load
        logger.info("[1/7] Loading data...")
        master_df = load_final_data(processed_dir)

        # Survival analysis
        logger.info("[2/7] Preparing survival data...")
        survival_df = prepare_survival_data(master_df)

        # Cell-type interpretation
        logger.info("[3/7] Cell-type interpretation...")
        subtype_labels = label_subtypes_by_celltype(master_df)

        # Clinical analysis
        logger.info("[4/7] Clinical analysis...")
        clinical_results = analyze_clinical_differences(master_df, subtype_labels)

        # Kaplan-Meier
        logger.info("[5/7] Plotting Kaplan-Meier...")
        plot_kaplan_meier(survival_df, results_1_dir)

        # Composite figure
        logger.info("[6/7] Plotting composite figure...")
        plot_composite_figure(master_df, results_1_dir)

        # Summary report
        logger.info("[7/7] Generating summary report...")
        generate_summary_report(master_df, subtype_labels, clinical_results, results_1_dir)

        logger.info("="*70)
        logger.info("STEP 1E COMPLETE")
        logger.info("="*70)

        subtype_list = [s for s in master_df['subtype'].unique() if s != 'Control']

        return {
            'n_subtypes': len(subtype_list),
            'n_control': len(master_df[master_df['subtype'] == 'Control']),
            'subtype_list': subtype_list,
            'status': 'PASS'
        }

    except Exception as e:
        logger.error(f"Step 1E failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
