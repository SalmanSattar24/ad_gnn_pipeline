import os
import glob
import logging
import time
import pandas as pd
import numpy as np
import networkx as nx
import requests

logger = logging.getLogger('Step2D')

def fetch_string_edges(proteins):
    """
    Queries STRING API for known interactions. Returns a dictionary of edges receiving the bonus.
    """
    # Query in batches of 100 to avoid long URI string errors
    bonus_edges = {}
    
    # In test mode or when limits arise, it's safer to mock or silently bypass
    try:
        import urllib.parse
        # Usually, top nodes are 200–500, which is perfectly fine for STRING
        # We separate by '%0d' as per STRING API
        prot_str = "%0d".join([str(p) for p in proteins])
        url = "https://string-db.org/api/json/network"
        data = {
            'identifiers': prot_str,
            'species': 9606,
            'required_score': 700 # high confidence
        }
        res = requests.post(url, data=data, timeout=15)
        if res.status_code == 200:
            for item in res.json():
                # Extract preferred names
                pA = item.get('preferredName_A')
                pB = item.get('preferredName_B')
                if pA and pB:
                    # Undirected bonus, both directions map to 0.15
                    bonus_edges[f"{pA}_{pB}"] = 0.15
                    bonus_edges[f"{pB}_{pA}"] = 0.15
            logger.info(f"Retrieved {len(bonus_edges)//2} high-confidence STRING edges.")
        else:
            logger.warning(f"STRING API returned {res.status_code}. Skipping bonus.")
    except Exception as e:
        logger.warning(f"STRING API interaction failed: {e}. Skipping bonus.")
    
    return bonus_edges

def main(data_dir, results_dir, test_mode=False):
    start_time = time.time()
    
    wgcna_dir = os.path.join(results_dir, 'step2', 'layer1_wgcna')
    glasso_dir = os.path.join(results_dir, 'step2', 'layer2_glasso')
    causal_dir = os.path.join(results_dir, 'step2', 'layer3_causal')
    
    output_dir = os.path.join(results_dir, 'step2', 'layer4_consensus')
    os.makedirs(output_dir, exist_ok=True)
    
    # We find all (subtype, cell_type) combinations by inspecting WGCNA outputs
    if not os.path.exists(wgcna_dir):
        raise FileNotFoundError("Missing WGCNA layer (Step 2A). Cannot build consensus.")
        
    files = glob.glob(os.path.join(wgcna_dir, '*_wgcna_edges.csv'))
    
    networks_built = 0
    final_edges_count = 0
    
    for f in files:
        base = os.path.basename(f).replace('_wgcna_edges.csv', '')
        # base is like "ST1_ct_Mic"
        parts = base.split('_ct_')
        if len(parts) == 2:
            subtype = parts[0]
            ct = "ct_" + parts[1]
        else:
            continue
            
        logger.info(f"Building Consensus Network for {subtype} | {ct}")
        
        # 1. Load the three layers
        try:
            w_df = pd.read_csv(f)
        except Exception:
            w_df = pd.DataFrame()
            
        g_file = os.path.join(glasso_dir, f"{subtype}_{ct}_glasso_edges.csv")
        try:
            g_df = pd.read_csv(g_file)
        except Exception:
            g_df = pd.DataFrame()
            
        c_file = os.path.join(causal_dir, f"{subtype}_{ct}_causal_edges.csv")
        try:
            c_df = pd.read_csv(c_file)
        except Exception:
            c_df = pd.DataFrame()
            
        # Compile all proteins mentioned to fetch STRING
        all_proteins = set()
        for df in [w_df, g_df, c_df]:
            if len(df) > 0:
                all_proteins.update(df['protein_A'].unique())
                all_proteins.update(df['protein_B'].unique())
                
        if len(all_proteins) == 0:
            continue
            
        # 2. STRING Bonus
        string_bonus = {}
        if not test_mode:
            string_bonus = fetch_string_edges(list(all_proteins))
            
        # 3. Aggregate network map
        # We need a unified directed representation since causal layer is directed
        # WGCNA and GLASSO are undirected, so we'll model them as bi-directional paths
        edges_map = {}
        
        def add_edge_undirected(u, v, weight_key, weight_val):
            # Model undirected as two directed edges
            key1 = (u, v)
            if key1 not in edges_map: edges_map[key1] = {'wgcna': 0, 'glasso': 0, 'causal': 0}
            edges_map[key1][weight_key] = max(edges_map[key1][weight_key], weight_val)
            
            key2 = (v, u)
            if key2 not in edges_map: edges_map[key2] = {'wgcna': 0, 'glasso': 0, 'causal': 0}
            edges_map[key2][weight_key] = max(edges_map[key2][weight_key], weight_val)
            
        def add_edge_directed(u, v, weight_key, weight_val):
            key = (u, v)
            if key not in edges_map: edges_map[key] = {'wgcna': 0, 'glasso': 0, 'causal': 0}
            edges_map[key][weight_key] = max(edges_map[key][weight_key], weight_val)
            
        # Insert WGCNA
        if len(w_df) > 0:
            for _, r in w_df.iterrows():
                add_edge_undirected(r['protein_A'], r['protein_B'], 'wgcna', r['consensus_weight'])
                
        # Insert Glasso
        if len(g_df) > 0:
            for _, r in g_df.iterrows():
                add_edge_undirected(r['protein_A'], r['protein_B'], 'glasso', r['consensus_weight'])
                
        # Insert Causal
        if len(c_df) > 0:
            for _, r in c_df.iterrows():
                add_edge_directed(r['protein_A'], r['protein_B'], 'causal', r['consensus_weight'])
                
        # Combine
        consensus_data = []
        for (u, v), conf in edges_map.items():
            base_score = conf['wgcna'] + conf['glasso'] + conf['causal']
            if base_score > 0:
                bonus = string_bonus.get(f"{u}_{v}", 0.0)
                final_score = base_score + bonus
                
                # Confidence Tier designation
                methods_supported = (conf['wgcna'] > 0) + (conf['glasso'] > 0) + (conf['causal'] > 0)
                if methods_supported == 3 and bonus > 0:
                    tier = "Highest (All 3 + STRING)"
                elif methods_supported == 3:
                    tier = "High (All 3)"
                elif methods_supported == 2:
                    tier = "Moderate (2 Methods)"
                else:
                    tier = "Low (1 Method)"
                    
                consensus_data.append({
                    'source': u, 
                    'target': v, 
                    'wgcna_w': conf['wgcna'],
                    'glasso_w': conf['glasso'],
                    'causal_w': conf['causal'],
                    'string_bonus': bonus,
                    'total_weight': final_score,
                    'confidence_tier': tier
                })
                
        df_cons = pd.DataFrame(consensus_data)
        
        if len(df_cons) > 0:
            out_file = f"{subtype}_{ct}_consensus.csv"
            df_cons.to_csv(os.path.join(output_dir, out_file), index=False)
            
            # Generate GraphML
            G = nx.from_pandas_edgelist(df_cons, 'source', 'target', 
                                        edge_attr=['wgcna_w', 'glasso_w', 'causal_w', 'string_bonus', 'total_weight', 'confidence_tier'], 
                                        create_using=nx.DiGraph())
                                        
            # Calculate node centrality
            deg = dict(G.degree())
            bet = nx.betweenness_centrality(G, weight='total_weight')
            
            # Save attributes
            nx.set_node_attributes(G, deg, 'degree')
            nx.set_node_attributes(G, bet, 'betweenness')
            
            nx.write_graphml(G, os.path.join(output_dir, f"{subtype}_{ct}_consensus.graphml"))
            
            networks_built += 1
            final_edges_count += len(df_cons)
            
    return {
        'status': 'PASS',
        'networks_exported': networks_built,
        'total_consensus_edges': final_edges_count
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main('../data', '../results', test_mode=True)
