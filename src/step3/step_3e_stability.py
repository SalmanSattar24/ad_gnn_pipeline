import torch
import numpy as np
import pandas as pd
import os
import logging
import time
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from step3.step_3b_model import GATv2Model
from step3.step_3c_trainer import train_model, evaluate
from step3.step_3d_explain import explain_predictions

logger = logging.getLogger('Step3E')

def calculate_jaccard(subset1, subset2):
    set1 = set(subset1)
    set2 = set(subset2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def run_stability_protocol(data_list, proteins, device, n_seeds=30, test_mode=False):
    """
    Orchestrates the 30-seed bootstrap stability protocol.
    """
    if test_mode:
        n_seeds = 2   # Fast: just verify the training loop runs
        epochs = 3    # Fast: just verify forward/backward pass works
        top_k = 5     # Smaller top-k set
    else:
        epochs = 50
        top_k = 20
        
    all_top_20 = []
    all_auroc = []
    
    n_samples = len(data_list)
    indices = np.arange(n_samples)
    
    for i in range(n_seeds):
        logger.info(f"Running Stability Seed {i+1}/{n_seeds}...")
        np.random.seed(i)
        np.random.shuffle(indices)
        
        # 70/30 Split
        split = int(0.7 * n_samples)
        train_idx = indices[:split]
        val_idx = indices[split:]
        
        train_data = [data_list[j] for j in train_idx]
        val_data = [data_list[j] for j in val_idx]
        
        train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
        
        # Initialize and Train
        model = GATv2Model(num_node_features=7).to(device)
        model = train_model(model, train_loader, val_loader, device, epochs=epochs)
        
        # Evaluate AUROC
        _, y_true, y_pred = evaluate(model, val_loader, device)
        try:
            auroc = roc_auc_score(y_true, y_pred)
            all_auroc.append(auroc)
        except:
            all_auroc.append(0.5)
            
        # Explain
        importance = explain_predictions(model, val_loader, device)
        
        # Extract Top proteins
        top_indices = np.argsort(importance)[-top_k:]
        top_proteins = [proteins[idx] for idx in top_indices if idx < len(proteins)]
        all_top_20.append(top_proteins)
        
    # Calculate Pairwise Jaccard
    jaccards = []
    for i in range(len(all_top_20)):
        for j in range(i + 1, len(all_top_20)):
            jaccards.append(calculate_jaccard(all_top_20[i], all_top_20[j]))
            
    mean_jaccard = np.mean(jaccards) if jaccards else 0
    mean_auroc = np.mean(all_auroc)
    
    # Identify Robust Proteins (in top-20 of >= 18 runs / 60%)
    threshold = int(0.6 * n_seeds)
    prot_counts = {}
    for top_list in all_top_20:
        for p in top_list:
            prot_counts[p] = prot_counts.get(p, 0) + 1
            
    stable_prots = [p for p, count in prot_counts.items() if count >= threshold]
    
    return {
        'mean_jaccard': mean_jaccard,
        'mean_auroc': mean_auroc,
        'stable_proteins': stable_prots,
        'protein_scores': prot_counts
    }

def main(data_dir, results_dir, test_mode=False):
    """
    Main Step 3E orchestrator. Loop over saved .pt data files.
    """
    processed_dir = os.path.join(results_dir, 'step3', 'processed_data')
    output_dir = os.path.join(results_dir, 'step3', 'stability_results')
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    pt_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
    
    results_summary = []
    
    # For matching proteins, we need to load any graph from Step 2
    consensus_dir = os.path.join(results_dir, 'step2', 'layer4_consensus')
    
    import networkx as nx
    
    for pf in pt_files:
        if test_mode and pf != pt_files[0]:
            continue
            
        logger.info(f"Analyzing Stability for {pf}")
        # weights_only=False is required for custom PyG Data objects in newer PyTorch versions
        data_bundle = torch.load(os.path.join(processed_dir, pf), weights_only=False)
        
        # Re-derive protein list from original graphml
        base = pf.replace('_pyg_data.pt', '')
        nw_file = os.path.join(consensus_dir, f"{base}_consensus.graphml")
        G = nx.read_graphml(nw_file)
        proteins = list(G.nodes)
        
        res = run_stability_protocol(data_bundle, proteins, device, test_mode=test_mode)
        
        # Save detailed scores
        res_df = pd.DataFrame(list(res['protein_scores'].items()), columns=['protein', 'frequency'])
        res_df['is_stable'] = res_df['protein'].isin(res['stable_proteins'])
        res_df.to_csv(os.path.join(output_dir, f"{base}_stability_scores.csv"), index=False)
        
        results_summary.append({
            'file': pf,
            'mean_jaccard': res['mean_jaccard'],
            'mean_auroc': res['mean_auroc'],
            'n_stable': len(res['stable_proteins'])
        })
        
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(os.path.join(output_dir, "step3_summary_table.csv"), index=False)
    
    return {
        'status': 'PASS',
        'subtypes_analyzed': len(results_summary),
        'summary': results_summary
    }
