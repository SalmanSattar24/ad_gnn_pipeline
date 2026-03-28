import os
import logging
import time
import pandas as pd
import numpy as np
from dask.distributed import Client, LocalCluster

try:
    from arboreto.algo import grnboost2
except ImportError:
    grnboost2 = None

logger = logging.getLogger('Step2C')

def compute_grn(df_matrix, test_mode=False):
    """
    Executes GRNBoost2 (fast GENIE3) to infer causal network edges.
    Returns: Edge list dataframe with top 5% edges.
    """
    if grnboost2 is None:
        logger.error("Arboreto missing. Cannot run Causal layer.")
        return pd.DataFrame(columns=['protein_A', 'protein_B', 'weight'])
        
    # Ensure columns are strings to prevent internal matching errors
    df_matrix.columns = df_matrix.columns.astype(str)
    proteins = list(df_matrix.columns)
    
    logger.info(f"Running GRNBoost2 ensemble on {df_matrix.shape} data...")
    
    # Create a custom conservative Client for Colab to prevent worker OOM/crash
    client = None
    cluster = None
    try:
        # Colab has limited RAM and cores
        cluster = LocalCluster(n_workers=2, threads_per_worker=1, memory_limit='3GB')
        client = Client(cluster)
        logger.info("  Custom Dask LocalCluster created for GRNBoost2.")
        
        network = grnboost2(expression_data=df_matrix, 
                            tf_names=proteins,
                            client_or_address=client)
    except Exception as e:
        logger.error(f"GRNBoost2 failed: {e}")
        if test_mode:
            return pd.DataFrame(columns=['protein_A', 'protein_B', 'weight'])
        raise e
    finally:
        if client is not None:
            client.close()
        if cluster is not None:
            cluster.close()

                        
    # Filter for top 5% importance scores as per Research Plan Table 2D
    n_total_edges = network.shape[0]
    n_keep = int(n_total_edges * 0.05)
    
    if n_keep == 0:
        return pd.DataFrame(columns=['protein_A', 'protein_B', 'weight'])
        
    network_top = network.sort_values(by='importance', ascending=False).head(n_keep)
    
    edges = network_top.copy()
    edges.rename(columns={'TF': 'protein_A', 'target': 'protein_B', 'importance': 'weight'}, inplace=True)
    
    edges['layer'] = 'causal'
    edges['consensus_weight'] = 0.35
    
    return edges

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
    
    output_dir = os.path.join(results_dir, 'step2', 'layer3_causal')
    os.makedirs(output_dir, exist_ok=True)
    
    edges_retained = 0
    generated_networks = []
    
    for subtype in subtypes:
        st_patients = subtype_patients[subtype_patients['subtype'] == subtype]['patient_id'].values
        
        st_deconv = deconv_df[deconv_df['patient_id'].isin(st_patients)]
        cell_types = st_deconv['cell_type'].unique()

        
        for ct in cell_types:
            ct_data = st_deconv[st_deconv['cell_type'] == ct]
            if len(ct_data) == 0:
                continue
                
            # Pivot to Patients x Proteins
            ct_matrix = ct_data.pivot(index='patient_id', columns='protein_id', values='abundance')

            ct_matrix.fillna(ct_matrix.mean(), inplace=True)
            
            # Optimization: Causal inference on 5000 proteins is extremely slow.
            # We filter for the top 500 most variable proteins per cell type.
            if not test_mode and ct_matrix.shape[1] > 500:
                variances = ct_matrix.var().sort_values(ascending=False)
                # Keep top 500 most variable proteins for high-signal causal modeling
                top_500 = variances.index[:500]
                ct_matrix = ct_matrix[top_500]
                logger.info(f"  Optimizing: filtered to top 500 proteins by variance for Causal Inference")

            if test_mode:
                # Use only top 15 proteins for fast test mode
                cols = ct_matrix.columns[:15]
                ct_matrix = ct_matrix[cols]
                
            logger.info(f"Running Causal Inference for {subtype} | {ct} (P={ct_matrix.shape[1]})")
            edges = compute_grn(ct_matrix, test_mode=test_mode)

            
            if len(edges) > 0:
                edges['subtype'] = subtype
                edges['cell_type'] = ct
                
                out_file = f"{subtype}_ct_{ct}_causal_edges.csv"
                edges.to_csv(os.path.join(output_dir, out_file), index=False)
                edges_retained += len(edges)
                generated_networks.append(f"{subtype}-{ct}")
                
    return {
        'status': 'PASS',
        'networks_generated': len(generated_networks),
        'total_causal_edges': edges_retained
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main('../data', '../results', test_mode=True)
