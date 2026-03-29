import os
import logging
import time
import pandas as pd
import numpy as np

logger = logging.getLogger('Step2A')

def pick_soft_threshold(corr_matrix, powers=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14], min_r2=0.85):
    """
    Selects the optimal soft-thresholding power beta for scale-free topology.
    """
    for beta in powers:
        adj = np.power(corr_matrix, beta)
        k = adj.sum(axis=1) - 1  # Node connectivity
        
        # Calculate scale-free topology fit: log10(p(k)) ~ log10(k)
        hist, bin_edges = np.histogram(k, bins=min(10, max(3, len(k)//5)), density=True)
        x = (bin_edges[:-1] + bin_edges[1:]) / 2
        y = hist
        
        valid = (y > 0) & (x > 0)
        if valid.sum() < 3:
            continue
            
        log_x = np.log10(x[valid])
        log_y = np.log10(y[valid])
        
        if len(log_x) > 1:
            r2 = np.corrcoef(log_x, log_y)[0, 1] ** 2
            if r2 >= min_r2:
                logger.info(f"Selected beta={beta} (R^2 = {r2:.3f})")
                return beta
                
    logger.info("Scale-free topology not reached above min_r2. Defaulting to beta=6.")
    return 6

def compute_tom(df_matrix, test_mode=False):
    """
    Compute Topological Overlap Measure (TOM) efficiently natively in numpy/pandas.
    """
    corr = df_matrix.corr(method='pearson').abs()
    
    if test_mode:
        beta = 4
    else:
        beta = pick_soft_threshold(corr.values)
        
    adj = np.power(corr, beta)
    
    # Connectivity
    k = adj.sum(axis=1)
    
    # Calculate l_ij efficiently using matrix multiplication
    l = np.dot(adj, adj)
    
    n_nodes = adj.shape[0]
    tom = np.zeros((n_nodes, n_nodes))
    
    k_vals = k.values
    adj_vals = adj.values
    
    # Vectorized TOM calculation
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            if i == j:
                tom[i, j] = 1.0
            else:
                min_k = min(k_vals[i], k_vals[j])
                val = (l[i, j] + adj_vals[i, j]) / (min_k + 1 - adj_vals[i, j])
                tom[i, j] = val
                tom[j, i] = val
                
    tom_df = pd.DataFrame(tom, index=adj.index, columns=adj.columns)
    return tom_df

def main(data_dir, results_dir, test_mode=False):
    """
    Executes Step 2A: WGCNA Co-expression network construction via TOM.
    Target output: Edge list of protein-protein co-expression.
    """
    start_time = time.time()
    
    # 1. Load Data
    master_df_path = os.path.join(data_dir, 'processed', 'master_patient_table_final.csv')
    deconv_path = os.path.join(data_dir, 'processed', 'deconvolved_profiles.csv')
    
    if not os.path.exists(master_df_path) or not os.path.exists(deconv_path):
        raise FileNotFoundError("Required Step 1 outputs missing.")
        
    master_df = pd.read_csv(master_df_path, index_col=0)
    # Ensure patient_id column exists
    if 'patient_id' not in master_df.columns:
        master_df['patient_id'] = master_df.index
    
    deconv_df = pd.read_csv(deconv_path)
    # Standardize column name if it was old version
    if 'sample_id' in deconv_df.columns:
        deconv_df.rename(columns={'sample_id': 'patient_id'}, inplace=True)
    
    # Filter only AD subtypes (ignore NaN subtypes which are healthy/AsymAD)
    subtype_patients = master_df.dropna(subset=['subtype'])
    subtypes = subtype_patients['subtype'].unique()
    
    # 2. Results container
    output_dir = os.path.join(results_dir, 'step2', 'layer1_wgcna')
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Process per subtype and per cell-type
    # The pipeline specifies running WGCNA per subtype on deconvolved matrices.
    edges_retained = 0
    generated_networks = []
    
    for subtype in subtypes:
        logger.info(f"Processing WGCNA for {subtype}")
        st_patients = subtype_patients[subtype_patients['subtype'] == subtype]['patient_id'].values
        
        st_deconv = deconv_df[deconv_df['patient_id'].isin(st_patients)]
        cell_types = st_deconv['cell_type'].unique()
        
        # In test_mode, only process the first cell type to validate the plumbing
        if test_mode:
            cell_types = cell_types[:1]

        
        for ct in cell_types:
            ct_data = st_deconv[st_deconv['cell_type'] == ct]
            if len(ct_data) == 0:
                continue
                
            # Pivot to Patients x Proteins
            ct_matrix = ct_data.pivot(index='patient_id', columns='protein_id', values='abundance')

            ct_matrix.fillna(ct_matrix.mean(), inplace=True) 
            
            if test_mode:
                # Use very few proteins to ensure speed across all layers (10 is enough to test plumbing)
                ct_matrix = ct_matrix.iloc[:, :10]
            
            # WGCNA execution
            tom_df = compute_tom(ct_matrix, test_mode=test_mode)
            
            # Filter TOM > 0.1 (Research Plan threshold)
            # In test mode, lower threshold to 0.01 for plumbing verification
            tom_df.values[np.tril_indices_from(tom_df)] = 0 # keep upper triangle
            tom_df.index.name = 'protein_A'
            tom_df.columns.name = 'protein_B'
            edges = tom_df.unstack().reset_index()
            edges.columns = ['protein_A', 'protein_B', 'weight']
            
            thresh = 0.01 if test_mode else 0.1
            edges = edges[edges['weight'] > thresh]
            edges = edges[edges['protein_A'] != edges['protein_B']]
            
            if len(edges) > 0:
                edges['subtype'] = subtype
                edges['cell_type'] = ct
                edges['layer'] = 'wgcna'
                edges['consensus_weight'] = 0.30
                
                out_file = f"{subtype}_ct_{ct}_wgcna_edges.csv"
                edges.to_csv(os.path.join(output_dir, out_file), index=False)
                edges_retained += len(edges)
                generated_networks.append(f"{subtype}-{ct}")
                
    return {
        'status': 'PASS',
        'networks_generated': len(generated_networks),
        'total_wgcna_edges': edges_retained
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main('../data', '../results', test_mode=True)
