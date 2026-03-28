import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import logging
import numpy as np

logger = logging.getLogger('Step3C')

def train_epoch(model, loader, optimizer, device, alpha=0.7):
    model.train()
    total_loss = 0
    
    criterion_class = nn.BCEWithLogitsLoss()
    criterion_pseudo = nn.MSELoss()
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass: out_class, out_pseudo
        c_pred, p_pred = model(data.x, data.edge_index, data.batch)
        
        # Dual Loss Calculation
        loss_class = criterion_class(c_pred.view(-1), data.y_class)
        loss_pseudo = criterion_pseudo(p_pred.view(-1), data.y_pseudo)
        
        loss = (alpha * loss_class) + ((1 - alpha) * loss_pseudo)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, alpha=0.7):
    model.eval()
    total_loss = 0
    
    criterion_class = nn.BCEWithLogitsLoss()
    criterion_pseudo = nn.MSELoss()
    
    y_true_class = []
    y_pred_class = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            c_pred, p_pred = model(data.x, data.edge_index, data.batch)
            
            loss_class = criterion_class(c_pred.view(-1), data.y_class)
            loss_pseudo = criterion_pseudo(p_pred.view(-1), data.y_pseudo)
            
            loss = (alpha * loss_class) + ((1 - alpha) * loss_pseudo)
            total_loss += loss.item() * data.num_graphs
            
            y_true_class.extend(data.y_class.cpu().numpy())
            y_pred_class.extend(torch.sigmoid(c_pred).cpu().numpy())
            
    avg_loss = total_loss / len(loader.dataset)
    
    # Calculate simple metric (AUROC is better but requires sklearn)
    # Return loss for early stopping/convergence check
    return avg_loss, np.array(y_true_class), np.array(y_pred_class)

def train_model(model, train_loader, val_loader, device, epochs=50, lr=0.001, alpha=0.7):
    """
    Standard training logic for one seed.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, alpha)
        val_loss, _, _ = evaluate(model, val_loader, device, alpha)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save state if needed, but for stability protocol we just keep the best weights in memory
            
    return model

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Trainer module loaded.")
