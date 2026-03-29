import torch
import numpy as np
import pandas as pd
import os
import logging
import time
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    # Fallback if xgboost is missing from the environment
    XGBClassifier, XGBRegressor = None, None

logger = logging.getLogger('Step3F')

def prepare_tabular_data(data_list):
    """
    Converts PyG Data objects into tabular features for baseline models.
    Strategy: Mean-aggregation of node features to graph-level.
    """
    X = []
    y_class = []
    y_pseudo = []
    
    for data in data_list:
        # data.x shape: [n_nodes, 7]
        # Aggregate to [7] features using mean
        graph_feat = data.x.mean(dim=0).numpy()
        X.append(graph_feat)
        y_class.append(data.y_class.item())
        y_pseudo.append(data.y_pseudo.item())
        
    return np.array(X), np.array(y_class), np.array(y_pseudo)

def run_benchmarking(data_bundle, base_name, output_dir, test_mode=False):
    """
    Trains and evaluates MLP and XGBoost baselines.
    """
    # 1. Prepare Data
    n_samples = len(data_bundle)
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    split = int(0.7 * n_samples)
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_data = [data_bundle[i] for i in train_idx]
    val_data = [data_bundle[i] for i in val_idx]
    
    X_train, yc_train, yp_train = prepare_tabular_data(train_data)
    X_val, yc_val, yp_val = prepare_tabular_data(val_data)
    
    results = []
    
    # --- MLP BASELINE ---
    logger.info("  Training MLP Baselines...")
    mlp_c = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200 if not test_mode else 10)
    mlp_r = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200 if not test_mode else 10)
    
    mlp_c.fit(X_train, yc_train)
    mlp_r.fit(X_train, yp_train)
    
    try:
        yc_pred_mlp = mlp_c.predict_proba(X_val)[:, 1]
        auc_mlp = roc_auc_score(yc_val, yc_pred_mlp)
    except:
        auc_mlp = 0.5
        
    yp_pred_mlp = mlp_r.predict(X_val)
    mse_mlp = mean_squared_error(yp_val, yp_pred_mlp)
    
    results.append({
        'model': 'MLP',
        'classification_auroc': auc_mlp,
        'regression_mse': mse_mlp
    })
    
    # --- XGBoost BASELINE ---
    if XGBClassifier is not None:
        logger.info("  Training XGBoost Baselines...")
        xgb_c = XGBClassifier(n_estimators=100 if not test_mode else 5, max_depth=3)
        xgb_r = XGBRegressor(n_estimators=100 if not test_mode else 5, max_depth=3)
        
        xgb_c.fit(X_train, yc_train)
        xgb_r.fit(X_train, yp_train)
        
        try:
            yc_pred_xgb = xgb_c.predict_proba(X_val)[:, 1]
            auc_xgb = roc_auc_score(yc_val, yc_pred_xgb)
        except:
            auc_xgb = 0.5
            
        yp_pred_xgb = xgb_r.predict(X_val)
        mse_xgb = mean_squared_error(yp_val, yp_pred_xgb)
        
        results.append({
            'model': 'XGBoost',
            'classification_auroc': auc_xgb,
            'regression_mse': mse_xgb
        })
    else:
        logger.warning("  XGBoost not installed. Skipping XGBoost baseline.")
        
    # Save Results
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, f"{base_name}_benchmarking.csv"), index=False)
    
    return results

def main(data_dir, results_dir, test_mode=False):
    processed_dir = os.path.join(results_dir, 'step3', 'processed_data')
    output_dir = os.path.join(results_dir, 'step3', 'benchmarking')
    os.makedirs(output_dir, exist_ok=True)
    
    pt_files = [f for f in os.listdir(processed_dir) if f.endswith('.pt')]
    
    bench_summary = []
    
    for pf in pt_files:
        if test_mode and pf != pt_files[0]:
            continue
            
        logger.info(f"Running Benchmarks for {pf}")
        data_bundle = torch.load(os.path.join(processed_dir, pf), weights_only=False)
        
        base = pf.replace('_pyg_data.pt', '')
        res = run_benchmarking(data_bundle, base, output_dir, test_mode=test_mode)
        
        for r in res:
            r['file'] = pf
            bench_summary.append(r)
            
    summary_df = pd.DataFrame(bench_summary)
    summary_df.to_csv(os.path.join(output_dir, "benchmarking_summary.csv"), index=False)
    
    return {
        'status': 'PASS',
        'benchmarks_run': len(bench_summary),
        'summary': bench_summary
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main('../data', '../results', test_mode=True)
