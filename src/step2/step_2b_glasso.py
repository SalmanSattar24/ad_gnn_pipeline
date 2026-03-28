import os
import logging
import time
import pandas as pd
import numpy as np
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore', category=UserWarning) # ignore glasso convergence warnings

logger = logging.getLogger('Step2B')

def fit_single_bootstrap(indices, data_matrix, lambdas):
    """
    Helper function for parallel bootstrap runs.
    """
    n_features = data_matrix.shape[1]
    sub_data = data_matrix[indices, :]
    
    # Center and scale
    sub_data = (sub_data - sub_data.mean(axis=0)) / (sub_data.std(axis=0) + 1e-8)
    emp_cov = np.cov(sub_data, rowvar=False)
    
    adjs = []
    for lam in lambdas:
        try:
            # High penalty/fast convergence for stability selection
            gl = GraphicalLasso(alpha=lam, max_iter=100, assume_centered=True, tol=1e-3)
            gl.fit(emp_cov)
            adj = (np.abs(gl.precision_) > 1e-5).astype(float)
            adjs.append(adj)
        except Exception:
            adjs.append(np.zeros((n_features, n_features)))
            
    return np.stack(adjs)

def compute_stars_lambda(data_matrix, n_subsamples=20, subsample_ratio=0.8, beta_star=0.05):
    """
    Accelerated Python implementation of StARS using joblib parallelization.
    """
    n_samples, n_features = data_matrix.shape
    subsample_size = int(n_samples * subsample_ratio)
    
    lambdas = np.logspace(0, -2.5, 10) # 10 lambdas
    
    logger.info(f"  Starting parallel StARS with {n_subsamples} subsamples...")
    
    # Prepare bootstrap indices
    bootstrap_indices = [np.random.choice(n_samples, size=subsample_size, replace=False) 
                         for _ in range(n_subsamples)]
    
    # Run in parallel across all available cores
    n_jobs = -1 # Use all cores 
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_single_bootstrap)(indices, data_matrix, lambdas) 
        for indices in bootstrap_indices
    )
    
    # Aggregate edge probabilities
    # results is list of (n_lambdas, n_features, n_features)
    edge_probs = np.sum(results, axis=0) / n_subsamples
    
    instabilities = 2 * edge_probs * (1 - edge_probs)
    
    # Calculate average instability per lambda for off-diagonal edges
    mask = ~np.eye(n_features, dtype=bool)
    avg_instability = np.zeros(len(lambdas))
    
    for idx in range(len(lambdas)):
        instab_matrix = instabilities[idx]
        avg_instability[idx] = np.mean(instab_matrix[mask])
        
    # Selected lambda: largest lambda s.t. instability <= beta_star
    selected_lambda = lambdas[0]
    for lam, instab in zip(lambdas, avg_instability):
        if instab <= beta_star:
            selected_lambda = lam
        else:
            break

    logger.info(f"  StARS selected lambda: {selected_lambda:.4f} (Instability: {instab:.4f})")
    return selected_lambda


def run_glasso(df_matrix, n_patients, test_mode=False):
    """
    Dispatches the correct GLASSO method based on n_patients.
    Returns: Edge list dataframe.
    """
    if n_patients < 40 and not test_mode:
        logger.info(f"Skipping GLASSO: N={n_patients} is < 40.")
        return pd.DataFrame(columns=['protein_A', 'protein_B', 'weight'])
        
    data = df_matrix.values
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    emp_cov = np.cov(data, rowvar=False)
    
    method_used = ""
    # According to plan:
    # >= 70 : Extended BIC / CV
    # 40-69 : StARS
    if test_mode:
        logger.info(f"Using fixed lambda for TEST MODE (N={n_patients})")
        method_used = "TestFixed"
        gl = GraphicalLasso(alpha=0.5, max_iter=20)
        gl.fit(emp_cov)
        prec = gl.precision_
    elif n_patients >= 70:
        logger.info(f"Using GraphicalLassoCV (N={n_patients} >= 70)")
        method_used = "CV"
        gl = GraphicalLassoCV(cv=5, max_iter=200)
        gl.fit(data)
        prec = gl.precision_
    else:
        logger.info(f"Using StARS Bootstrapping (N={n_patients})")
        method_used = "StARS"
        lam = compute_stars_lambda(data, n_subsamples=20)

        gl = GraphicalLasso(alpha=lam, max_iter=200)
        gl.fit(emp_cov)
        prec = gl.precision_
        
    # Extract edges from precision matrix
    # Non-zero precision elements imply conditional dependence
    n_features = prec.shape[0]
    proteins = df_matrix.columns
    
    edges = []
    # Only keep upper triangle
    for i in range(n_features):
        for j in range(i+1, n_features):
            val = prec[i, j]
            if abs(val) > 1e-5:
                edges.append({'protein_A': proteins[i], 'protein_B': proteins[j], 'weight': abs(val)})
                
    df_edges = pd.DataFrame(edges)
    if len(df_edges) > 0:
        df_edges['layer'] = 'glasso'
        df_edges['consensus_weight'] = 0.35
        df_edges['glasso_method'] = method_used
    return df_edges

def main(data_dir, results_dir, test_mode=False):
    start_time = time.time()
    
    master_df_path = os.path.join(data_dir, 'processed', 'master_patient_table_final.csv')
    deconv_path = os.path.join(data_dir, 'processed', 'deconvolved_profiles.csv')
    
    master_df = pd.read_csv(master_df_path, index_col=0)
    # Ensure patient_id column exists
    if 'patient_id' not in master_df.columns:
        master_df['patient_id'] = master_df.index
        
    deconv_df = pd.read_csv(deconv_path)
    # Standardize column name if it was old version
    if 'sample_id' in deconv_df.columns:
        deconv_df.rename(columns={'sample_id': 'patient_id'}, inplace=True)
    
    subtype_patients = master_df.dropna(subset=['subtype'])
    subtypes = subtype_patients['subtype'].unique()
    
    output_dir = os.path.join(results_dir, 'step2', 'layer2_glasso')
    os.makedirs(output_dir, exist_ok=True)
    
    edges_retained = 0
    generated_networks = []
    
    for subtype in subtypes:
        st_patients = subtype_patients[subtype_patients['subtype'] == subtype]['patient_id'].values
        n_patients = len(st_patients)
        
        st_deconv = deconv_df[deconv_df['patient_id'].isin(st_patients)]
        cell_types = st_deconv['cell_type'].unique()

        
        for ct in cell_types:
            ct_data = st_deconv[st_deconv['cell_type'] == ct]
            if len(ct_data) == 0:
                continue
                
            ct_matrix = ct_data.pivot(index='patient_id', columns='protein_id', values='abundance')

            ct_matrix.fillna(ct_matrix.mean(), inplace=True)
            
            # Optimization: GLASSO is O(P^3). For 5000 proteins, this will never finish.
            # We filter for the top 500 most variable proteins per cell type.
            if not test_mode and ct_matrix.shape[1] > 500:
                variances = ct_matrix.var().sort_values(ascending=False)
                top_500 = variances.index[:500]
                ct_matrix = ct_matrix[top_500]
                logger.info(f"  Optimizing: filtered to top 500 proteins by variance for GLASSO")

            if test_mode:
                ct_matrix = ct_matrix.iloc[:, :15]
            
            logger.info(f"Running GLASSO for {subtype} | {ct} (N={n_patients}, P={ct_matrix.shape[1]})")
            edges = run_glasso(ct_matrix, n_patients, test_mode=test_mode)

            
            if len(edges) > 0:
                edges['subtype'] = subtype
                edges['cell_type'] = ct
                
                out_file = f"{subtype}_ct_{ct}_glasso_edges.csv"
                edges.to_csv(os.path.join(output_dir, out_file), index=False)
                edges_retained += len(edges)
                generated_networks.append(f"{subtype}-{ct}")
                
    return {
        'status': 'PASS',
        'networks_generated': len(generated_networks),
        'total_glasso_edges': edges_retained
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main('../data', '../results', test_mode=True)
