import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GATv2Model(torch.nn.Module):
    """
    Graph Attention Network version 2 (GATv2) for AD Subtype prediction.
    Features a dual-head output for classification and regression.
    """
    def __init__(self, num_node_features, hidden_channels=32, heads=4, dropout=0.2):
        super(GATv2Model, self).__init__()
        
        # 1. First GATv2 Layer
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=heads, dropout=dropout)
        
        # 2. Second GATv2 Layer
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        
        # 3. Dense layers for multi-objective output
        # Global mean pooling will reduce graph to [1, hidden_channels]
        
        # Head A: Subtype Classification (Binary)
        self.class_head = Linear(hidden_channels, 1)
        
        # Head B: Pseudotime Regression (Continuous)
        self.pseudo_head = Linear(hidden_channels, 1)
        
        self.dropout = Dropout(p=dropout)

    def forward(self, x, edge_index, batch):
        # 1. Graph Convolution 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        # 2. Graph Convolution 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # 3. Global Pooling (Convert node-level to graph-level representation)
        x = global_mean_pool(x, batch)
        
        # 4. Final MLP Layers
        # Classification Head
        out_class = self.class_head(x)
        
        # Regression Head
        out_pseudo = self.pseudo_head(x)
        
        return out_class, out_pseudo

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # Test model instantiation
    model = GATv2Model(num_node_features=7)
    print(f"Model instantiated with {count_parameters(model)} trainable parameters.")
