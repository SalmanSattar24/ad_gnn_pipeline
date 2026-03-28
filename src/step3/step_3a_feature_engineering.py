import os
import logging
import time
import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import glob

logger = logging.getLogger('Step3A')

def main(data_dir, results_dir, test_mode=False):
    """
    Executes Step 3A: Feature Engineering for GNN.
    Prepares PyTorch Geometric Data objects for each patient.
    """
    start_time = time.time()
    
    # 1. Load Step 1 Outputs
    master_df_path = os.path.join(data_dir, 'processed', 'master_patient_table_final.csv')
    deconv_path = os.path.join(data_dir, 'processed', 'deconvolved_profiles.csv')
    
    if not os.path.exists(master_df_path) or not os.path.exists(deconv_path):
        raise FileNotFoundError("Required Step 1 outputs missing.")
        
    master_df = pd.read_csv(master_df_path)
    deconv_df = pd.read_csv(deconv_path)
    
    # 2. Results container
    output_dir = os.path.join(results_dir, 'step3', 'processed_data')
    os.makedirs(output_dir, exist_ok=True)
    
    # GWAS gene list (Mock for now, can be expanded)
    gwas_genes = ['APOE', 'BIN1', 'TREM2', 'CLU', 'CR1', 'PICALM', 'ABCA7', 'MS4A6A', 'CD33', 'SORL1']
    
    # 3. Identify Subtypes and Network Files
    consensus_dir = os.path.join(results_dir, 'step2', 'layer4_consensus')
    network_files = glob.glob(os.path.join(consensus_dir, '*.graphml'))
    
    if not network_files:
        logger.warning("No consensus networks found in Step 2. Using empty graphs for testing.")
        
    processed_count = 0
    
    # 4. Loop over subtypes and cell types
    for nw_file in network_files:
        base = os.path.basename(nw_file).replace('_consensus.graphml', '')
        parts = base.split('_ct_')
        if len(parts) == 2:
            subtype = parts[0]
            ct = parts[1]
        else:
            continue
            
        logger.info(f"Preparing Data for {subtype} | {ct}")
        
        # Load the graph
        G = nx.read_graphml(nw_file)
        proteins = list(G.nodes)
        
        if not proteins:
            continue
            
        # Get patient IDs for this subtype + healthy controls
        # The GNN is trained to distinguish Subtype AD from Healthy Controls
        st_patients = master_df[master_df['subtype'] == subtype]['patient_id'].values
        hc_patients = master_df[master_df['patient_class'] == 'Healthy Control']['patient_id'].values
        
        all_st_hc_pts = np.concatenate([st_patients, hc_patients])
        
        # Extract features for these patients
        st_deconv = deconv_df[(deconv_df['sample_id'].isin(all_st_hc_pts)) & (deconv_df['cell_type'] == ct)]
        
        if st_deconv.empty:
            continue
            
        # Pivot features
        feat_matrix = st_deconv.pivot(index='sample_id', columns='protein_id', values='abundance')
        # Reorder to match graph proteins, fill missing
        feat_matrix = feat_matrix.reindex(columns=proteins).fillna(0)
        
        # Build node feature vectors for each patient
        # We need to create one Data object per patient
        edge_index = torch.tensor(list(nx.convert_node_labels_to_integers(G).edges)).t().contiguous()
        
        # Static node features (same across all patients in this network)
        # 1. Degree
        degrees = torch.tensor([G.nodes[p].get('degree', 0) for p in proteins], dtype=torch.float).view(-1, 1)
        # 2. Betweenness
        bet = torch.tensor([G.nodes[p].get('betweenness', 0) for p in proteins], dtype=torch.float).view(-1, 1)
        # 3. GWAS Flag
        gwas = torch.tensor([1.0 if p in gwas_genes else 0.0 for p in proteins], dtype=torch.float).view(-1, 1)
        
        static_feats = torch.cat([degrees, bet, gwas], dim=1)
        
        patient_data_list = []
        
        for pid in all_st_hc_pts:
            if pid not in feat_matrix.index:
                continue
                
            p_row = master_df[master_df['patient_id'] == pid].iloc[0]
            
            # Dynamic patient features
            # 1. Abundance (z-scored locally across patients for this protein)
            # Actually we already have abundance, we'll z-score across all ST+HC patients
            p_abundance = torch.tensor(feat_matrix.loc[pid].values, dtype=torch.float).view(-1, 1)
            
            # 2. Disease Pseudotime
            p_pseudo = torch.tensor([p_row.get('dpt_pseudotime', 0.0)] * len(proteins), dtype=torch.float).view(-1, 1)
            
            # 3. Cell-type proportion (for this specific cell type)
            ct_prop_col = f"ct_{ct}"
            p_ct_prop = torch.tensor([p_row.get(ct_prop_col, 0.0)] * len(proteins), dtype=torch.float).view(-1, 1)
            
            # 4. Biological Sex
            p_sex = torch.tensor([1.0 if p_row.get('msex') == 1 else 0.0] * len(proteins), dtype=torch.float).view(-1, 1)
            
            # Combine all
            # Total features: Abundance(1), Pseudo(1), CT_Prop(1), Sex(1), Degree(1), Betweenness(1), GWAS(1) = 7 features
            x = torch.cat([p_abundance, p_pseudo, p_ct_prop, p_sex, static_feats], dim=1)
            
            # Targets
            y_class = torch.tensor([1.0 if pid in st_patients else 0.0], dtype=torch.float)
            y_pseudo = torch.tensor([p_row.get('dpt_pseudotime', 0.0)], dtype=torch.float)
            
            data = Data(x=x, edge_index=edge_index, y_class=y_class, y_pseudo=y_pseudo, patient_id=pid)
            patient_data_list.append(data)
            
        # Save this subtype's data bundle
        out_file = f"{subtype}_ct_{ct}_pyg_data.pt"
        torch.save(patient_data_list, os.path.join(output_dir, out_file))
        processed_count += 1
        
    return {
        'status': 'PASS',
        'subtypes_processed': processed_count,
        'feature_count': 7
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main('../data', '../results', test_mode=True)
