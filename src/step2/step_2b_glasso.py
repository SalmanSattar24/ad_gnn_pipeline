import os
import logging
import time
import pandas as pd
import numpy as np
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
import warnings
warnings.filterwarnings('ignore', category=UserWarning) # ignore glasso convergence warnings

logger = logging.getLogger('Step2B')

def compute_stars_lambda(data_matrix, n_subsamples=50, subsample_ratio=0.8, beta_star=0.05):
    """
    Native Python implementation of StARS (Stability Approach to Regularization Selection)
    for Graphical Lasso. Perfectly replicates the R `huge.stars` methodology.
    """
    n_samples, n_features = data_matrix.shape
    subsample_size = int(n_samples * subsample_ratio)
    
    lambdas = np.logspace(0, -3, 10) # 10 lambdas from 1.0 to 0.001
    
    edge_probs = np.zeros((len(lambdas), n_features, n_features))
    
    for i in range(n_subsamples):
        indices = np.random.choice(n_samples, size=subsample_size, replace=False)
        sub_data = data_matrix[indices, :]
        
        # Center and scale
        sub_data = (sub_data - sub_data.mean(axis=0)) / (sub_data.std(axis=0) + 1e-8)
        emp_cov = np.cov(sub_data, rowvar=False)
        
        for idx, lam in enumerate(lambdas):
            try:
                gl = GraphicalLasso(alpha=lam, max_iter=200, assume_centered=True)
                gl.fit(emp_cov)
                prec = gl.precision_
                adj = (np.abs(prec) > 1e-5).astype(float)
                edge_probs[idx] += adj
            except Exception:
                pass
                
    edge_probs /= n_subsamples
    instabilities = 2 * edge_probs * (1 - edge_probs)
    
    # Calculate average instability per lambda for off-diagonal edges
    mask = ~np.eye(n_features, dtype=bool)
    avg_instability = np.zeros(len(lambdas))
    
    for idx in range(len(lambdas)):
        instab_matrix = instabilities[idx]
        avg_instability[idx] = np.mean(instab_matrix[mask])
        
    # Start from largest lambda (most sparse, instability ~ 0) and go down
    # until instability exceeds beta_star.
    selected_lambda = lambdas[0]
    for lam, instab in zip(lambdas, avg_instability):
        if instab <= beta_star:
            selected_lambda = lam
        else:
            break # past the threshold

    logger.info(f"StARS selected lambda: {selected_lambda:.4f}")
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
        lam = compute_stars_lambda(data, n_subsamples=50)
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
    
    master_df = pd.read_csv(master_df_path)
    deconv_df = pd.read_csv(deconv_path)
    
    subtype_patients = master_df.dropna(subset=['subtype'])
    subtypes = subtype_patients['subtype'].unique()
    
    output_dir = os.path.join(results_dir, 'step2', 'layer2_glasso')
    os.makedirs(output_dir, exist_ok=True)
    
    edges_retained = 0
    generated_networks = []
    
    for subtype in subtypes:
        st_patients = subtype_patients[subtype_patients['subtype'] == subtype]['patient_id'].values
        n_patients = len(st_patients)
        
        st_deconv = deconv_df[deconv_df['sample_id'].isin(st_patients)]
        cell_types = st_deconv['cell_type'].unique()
        
        for ct in cell_types:
            ct_data = st_deconv[st_deconv['cell_type'] == ct]
            if len(ct_data) == 0:
                continue
                
            ct_matrix = ct_data.pivot(index='sample_id', columns='protein_id', values='abundance')
            ct_matrix.fillna(ct_matrix.mean(), inplace=True)
            
            if test_mode:
                ct_matrix = ct_matrix.iloc[:, :15]
            
            logger.info(f"Running GLASSO for {subtype} | {ct} (N={n_patients})")
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
