"""
Step 1A: Data Loading & Preprocessing

Implements:
  - Auto-detection of matrix orientation
  - QC filtering (>50% missing proteins removed)
  - KNN imputation
  - Log2 transformation
  - Z-score normalization
  - Covariate regression (age, sex, PMI)
  - Synthetic data generation for testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import warnings
import os
import logging

warnings.filterwarnings('ignore')


def setup_synthetic_data(raw_data_dir, n_patients=180, n_proteins=5000, test_mode=False):
    """Generate synthetic raw data for testing."""
    if test_mode:
        n_patients = 50
        n_proteins = 200

    np.random.seed(42)
    os.makedirs(raw_data_dir, exist_ok=True)

    # Synthetic proteomics
    patient_ids = [f'patient_{i:03d}' for i in range(n_patients)]
    protein_ids = [f'PROT_{i:05d}' for i in range(n_proteins)]

    proteomics_data = np.random.exponential(scale=10, size=(n_patients, n_proteins))
    missing_rate = 0.15
    missing_idx = np.random.choice([True, False], size=(n_patients, n_proteins),
                                    p=[missing_rate, 1-missing_rate])
    proteomics_data[missing_idx] = np.nan

    proteomics_df = pd.DataFrame(proteomics_data, index=patient_ids, columns=protein_ids)
    proteomics_df.index.name = 'patient_id'

    # Synthetic metadata
    diagnoses = np.random.choice(['Control', 'AD'], size=n_patients, p=[0.55, 0.45])
    metadata_data = {
        'diagnosis': diagnoses,
        'age_death': np.random.randint(60, 100, n_patients),
        'msex': np.random.choice([0, 1], n_patients),
        'pmi': np.random.uniform(2, 30, n_patients)
    }

    metadata_df = pd.DataFrame(metadata_data, index=patient_ids)
    metadata_df.index.name = 'projid'

    # Save
    raw_proteomics_file = f"{raw_data_dir}/raw_proteomics.csv"
    raw_metadata_file = f"{raw_data_dir}/raw_metadata.csv"

    proteomics_df.to_csv(raw_proteomics_file)
    metadata_df.to_csv(raw_metadata_file)

    logging.info(f"[SETUP] Generated synthetic proteomics: {proteomics_df.shape}")
    logging.info(f"[SETUP] Generated synthetic metadata: {metadata_df.shape}")

    return raw_proteomics_file, raw_metadata_file


def load_proteomics_data(file_path):
    df = pd.read_csv(file_path, index_col=0)
    n_rows, n_cols = df.shape

    # Auto-detect matrix orientation
    if n_rows > n_cols and n_rows > 1000:
        logging.info(f"  Auto-detected proteins as rows, transposing...")
        df = df.T

    logging.info(f"  Loaded proteomics: {df.shape[0]} samples x {df.shape[1]} proteins")
    return df


def load_clinical_metadata(file_path):
    df = pd.read_csv(file_path, index_col=0)
    logging.info(f"  Loaded metadata: {df.shape}")
    return df


def join_proteomics_and_metadata(proteomics_df, metadata_df):
    common_samples = proteomics_df.index.intersection(metadata_df.index)
    proteomics_df = proteomics_df.loc[common_samples]
    metadata_df = metadata_df.loc[common_samples]
    logging.info(f"  Joined on {len(common_samples)} common samples")
    return proteomics_df, metadata_df


def apply_subsample_filter(proteomics_df, metadata_df, n_samples=50):
    """For test mode only: subsample to first n_samples."""
    indices = proteomics_df.index[:n_samples]
    logging.info(f"  Subsampling to {n_samples} patients (test mode)")
    return proteomics_df.loc[indices], metadata_df.loc[indices]


def print_summary_statistics(proteomics_df, metadata_df, logger):
    for diag_col in ['diagnosis', 'cogdx', 'Diagnosis']:
        if diag_col in metadata_df.columns:
            logger.info(f"Diagnosis: {metadata_df[diag_col].value_counts().to_dict()}")
            break

    missing_pct = (proteomics_df.isnull().sum() / len(proteomics_df) * 100)
    logger.info(f"Proteomics: {proteomics_df.shape[0]} samples x {proteomics_df.shape[1]} proteins")
    logger.info(f"Missing values: {proteomics_df.isnull().sum().sum()} "
                f"({(proteomics_df.isnull().sum().sum() / proteomics_df.size * 100):.2f}%)")
    return missing_pct


def qc_filter_proteins(proteomics_df, threshold=0.50):
    missing_pct = (proteomics_df.isnull().sum() / len(proteomics_df) * 100)
    proteins_before = len(proteomics_df.columns)
    proteins_to_keep = missing_pct[missing_pct <= (threshold * 100)].index
    proteomics_df = proteomics_df[proteins_to_keep]
    proteins_after = len(proteomics_df.columns)
    logging.info(f"  QC: removed {proteins_before - proteins_after} proteins (>50% missing)")
    return proteomics_df, missing_pct


def knn_impute_missing_values(proteomics_df, k=5):
    if proteomics_df.shape[1] == 0 or proteomics_df.isnull().sum().sum() == 0:
        return proteomics_df

    n_neighbors = min(k, proteomics_df.shape[0] - 1)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    proteomics_array = imputer.fit_transform(proteomics_df)
    proteomics_df = pd.DataFrame(proteomics_array, columns=proteomics_df.columns,
                                  index=proteomics_df.index)
    logging.info(f"  KNN imputation complete (k={n_neighbors})")
    return proteomics_df


def log2_transform(proteomics_df, pseudocount=1):
    proteomics_df = np.log2(proteomics_df + pseudocount)
    logging.info(f"  Log2 transformation complete")
    return proteomics_df


def zscore_normalize(proteomics_df):
    scaler = StandardScaler()
    proteomics_array = scaler.fit_transform(proteomics_df)
    proteomics_df = pd.DataFrame(proteomics_array, columns=proteomics_df.columns,
                                  index=proteomics_df.index)
    logging.info(f"  Z-score normalization complete")
    return proteomics_df


def regress_out_covariates(proteomics_df, metadata_df, covariate_cols=['age_death', 'msex', 'pmi']):
    covariates_available = {}
    for col in covariate_cols:
        variants = [col, col.lower(), col.upper()]
        for var in variants:
            if var in metadata_df.columns:
                covariates_available[col] = var
                break

    if not covariates_available:
        logging.info(f"  No covariates found in metadata, skipping regression")
        return proteomics_df

    X = metadata_df[[covariates_available[key] for key in covariates_available]].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    X = X.fillna(X.mean())

    proteomics_residuals = proteomics_df.copy()
    for protein in proteomics_df.columns:
        y = proteomics_df[protein].values
        model = LinearRegression()
        model.fit(X, y)
        residuals = y - model.predict(X)
        proteomics_residuals[protein] = residuals

    logging.info(f"  Covariate regression complete ({len(covariates_available)} covariates)")
    return proteomics_residuals


def save_processed_data(proteomics_df, metadata_df, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)

    proteomics_file = f"{processed_dir}/rosmap_proteomics_cleaned.csv"
    metadata_file = f"{processed_dir}/rosmap_metadata.csv"

    proteomics_df.to_csv(proteomics_file)
    metadata_df.to_csv(metadata_file)

    logging.info(f"  Saved: rosmap_proteomics_cleaned.csv ({proteomics_df.shape})")
    logging.info(f"  Saved: rosmap_metadata.csv ({metadata_df.shape})")

    return proteomics_file, metadata_file


def generate_qc_report(proteomics_df_before_qc, proteomics_df_after, metadata_df,
                       missing_pct_before, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(missing_pct_before, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(50, color='red', linestyle='--', linewidth=2, label='50% threshold')
    ax.set_xlabel('Missing Value Percentage (%)', fontsize=11)
    ax.set_ylabel('Number of Proteins', fontsize=11)
    ax.set_title('Missing Values per Protein (Before QC)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(proteomics_df_after)

    diagnosis_col = None
    for col in ['diagnosis', 'cogdx', 'Diagnosis']:
        if col in metadata_df.columns:
            diagnosis_col = col
            break

    if diagnosis_col:
        diagnosis = metadata_df[diagnosis_col].values
        unique_diagnoses = np.unique(diagnosis)
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_diagnoses)))
        color_map = {diag: colors[i] for i, diag in enumerate(unique_diagnoses)}
        for diag in unique_diagnoses:
            mask = diagnosis == diag
            ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1], label=str(diag),
                      alpha=0.6, s=50, color=color_map[diag])
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
        ax.set_title('PCA of Samples (After Preprocessing)', fontsize=12, fontweight='bold')
        ax.legend(title='Diagnosis', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    qc_file = f"{results_dir}/qc_report.png"
    plt.savefig(qc_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"  Saved: qc_report.png")


def main(data_dir="data", results_dir="results", test_mode=False):
    """
    Run Step 1A: Data loading & preprocessing.

    Returns:
        dict: Results summary with keys: n_samples, n_proteins, missing_pct
    """
    logger = logging.getLogger("Step1A")

    try:
        raw_data_dir = f"{data_dir}/raw"
        processed_dir = f"{data_dir}/processed"
        results_1_dir = f"{results_dir}/step1"

        logger.info("="*70)
        logger.info("STEP 1A: Data Loading & Preprocessing")
        logger.info("="*70)

        # Setup synthetic data
        raw_prot_file, raw_meta_file = setup_synthetic_data(raw_data_dir, test_mode=test_mode)

        # Load
        logger.info("[1/10] Loading proteomics...")
        proteomics_df = load_proteomics_data(raw_prot_file)
        proteomics_df_original = proteomics_df.copy()

        logger.info("[2/10] Loading metadata...")
        metadata_df = load_clinical_metadata(raw_meta_file)

        # Subsample if in test mode
        if test_mode:
            logger.info("[TEST] Subsampling to 50 patients")
            proteomics_df, metadata_df = apply_subsample_filter(proteomics_df, metadata_df, n_samples=50)

        # Join
        logger.info("[3/10] Joining data...")
        proteomics_df, metadata_df = join_proteomics_and_metadata(proteomics_df, metadata_df)

        # Summary
        logger.info("[4/10] Computing statistics...")
        missing_pct_before = print_summary_statistics(proteomics_df, metadata_df, logger)

        # QC filter
        logger.info("[5/10] QC filtering...")
        proteomics_df, missing_pct_before = qc_filter_proteins(proteomics_df, threshold=0.50)

        # Imputation
        logger.info("[6/10] KNN imputation...")
        proteomics_df = knn_impute_missing_values(proteomics_df, k=5)

        # Log2
        logger.info("[7/10] Log2 transformation...")
        proteomics_df = log2_transform(proteomics_df, pseudocount=1)

        # Z-score
        logger.info("[8/10] Z-score normalization...")
        proteomics_df = zscore_normalize(proteomics_df)

        # Covariate regression
        logger.info("[9/10] Covariate regression...")
        proteomics_df = regress_out_covariates(proteomics_df, metadata_df)

        # Save
        logger.info("[10/10] Saving outputs...")
        save_processed_data(proteomics_df, metadata_df, processed_dir)
        generate_qc_report(proteomics_df_original, proteomics_df, metadata_df,
                          missing_pct_before, results_1_dir)

        logger.info("="*70)
        logger.info("STEP 1A COMPLETE")
        logger.info("="*70)

        results = {
            'n_samples': proteomics_df.shape[0],
            'n_proteins': proteomics_df.shape[1],
            'missing_pct': proteomics_df.isnull().sum().sum() / proteomics_df.size * 100,
            'status': 'PASS'
        }

        return results

    except Exception as e:
        logger.error(f"Step 1A failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
