import torch
from captum.attr import IntegratedGradients
import numpy as np
import logging

logger = logging.getLogger('Step3D')

def explain_predictions(model, loader, device):
    """
    Computes node-level feature importance using Integrated Gradients.
    Returns: Average importance score per protein.
    """
    model.eval()
    
    # We want attributions with respect to the Class Head (Classification)
    # Since GATv2 is a graph-level model in our case, we attribute back to nodes
    
    def forward_wrapper(x, edge_index, batch):
        c_out, _ = model(x, edge_index, batch)
        return c_out
        
    ig = IntegratedGradients(forward_wrapper)
    
    total_attributions = None
    sample_count = 0
    
    for data in loader:
        data = data.to(device)
        
        # IG requires a baseline (usually zero)
        # We process patients individually for clean attribution
        # (Though IG can handle batches, we want node-level mapping)
        
        # For simplicity in this implementation, we process graphs 1 by 1
        # Each 'data' object in loader could have multiple graphs
        
        batch_list = data.to_data_list()
        
        for g_data in batch_list:
            x = g_data.x.clone().requires_grad_()
            edge_index = g_data.edge_index
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            
            # target=0 for binary classification head output
            attr = ig.attribute(x, target=0, 
                                additional_forward_args=(edge_index, batch),
                                internal_batch_size=1)
            
            # Map back to protein nodes (sum over node features)
            # attr shape: [num_nodes, num_features]
            node_importance = attr.abs().sum(dim=1).detach().cpu().numpy()
            
            if total_attributions is None:
                total_attributions = node_importance
            else:
                total_attributions += node_importance
                
            sample_count += 1
            
    if total_attributions is not None and sample_count > 0:
        avg_importance = total_attributions / sample_count
        return avg_importance
    else:
        return np.array([])
