import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from tslearn.metrics import dtw
from itertools import combinations

def load_all_data(node_features_path, station_data_folder):
    """Load node features and time-series data"""
    node_features = pd.read_csv(node_features_path)
    station_data = {}
    
    for station in node_features['station'].unique():
        station_path = os.path.join(station_data_folder, station)
        if os.path.exists(station_path):
            components = {}
            for comp in ['HZ', 'HN', 'HE']:
                file_path = os.path.join(station_path, f"{station}_{comp}.csv")
                if os.path.exists(file_path):
                    components[comp] = pd.read_csv(file_path)
            station_data[station] = components
    
    return node_features, station_data

def calculate_geo_similarity(node_features: pd.DataFrame):
    """
    Compute geographical proximity between stations using lat/lon.
    Returns adjacency matrix and distance matrix.
    """
    lat = node_features["receiver_latitude"].values
    lon = node_features["receiver_longitude"].values
    km_per_degree = 111  # rough estimate

    # Compute dx and dy
    dx = np.outer(np.ones_like(lat), lon) - lon[:, np.newaxis]
    dx = dx * np.cos(np.radians(lat[:, np.newaxis])) * km_per_degree
    dy = (np.outer(np.ones_like(lat), lat) - lat[:, np.newaxis]) * km_per_degree

    geo_distances = np.sqrt(dx**2 + dy**2)

    geo_threshold = 50  # kilometers
    geo_adjacency = (geo_distances <= geo_threshold).astype(int)

    return geo_adjacency, geo_distances


def calculate_ts_similarity(station_data, node_features, sample_size=1000):
    """Calculate time-series similarity between stations focusing on waveform patterns
    
    Args:
        station_data: Dictionary containing time-series data for each station
        node_features: DataFrame with station metadata
        sample_size: Number of samples to use for comparison (default: 1000)
        
    Returns:
        ts_adjacency: Binary adjacency matrix (1=similar, 0=not similar)
        combined_similarity: Combined similarity scores matrix
        component_similarities: Dictionary of component-wise similarity matrices
    """
    stations = sorted(station_data.keys())  # Ensure consistent ordering
    n_stations = len(stations)
    component_similarities = {'HZ': np.zeros((n_stations, n_stations)),
                            'HN': np.zeros((n_stations, n_stations)),
                            'HE': np.zeros((n_stations, n_stations))}
    
    # Determine optimal sample size considering all components
    min_length = min(min(len(data[comp]) for data in station_data.values() )
                     for comp in ['HZ', 'HN', 'HE'])
    sample_size = min(sample_size, min_length)
    
    # Pre-calculate random indices for consistent sampling
    rng = np.random.default_rng(seed=42)  # For reproducible sampling
    sample_indices = rng.choice(min_length, size=sample_size, replace=False)
    
    for i, j in combinations(range(n_stations), 2):  # Only unique pairs
        st1, st2 = stations[i], stations[j]
        
        for comp in ['HZ', 'HN', 'HE']:
            # Get consistent samples using pre-selected indices
            ts1 = station_data[st1][comp].iloc[sample_indices]['acceleration'].values
            ts2 = station_data[st2][comp].iloc[sample_indices]['acceleration'].values
            
            # Robust normalization focusing on waveform shape
            def robust_normalize(x):
                x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-10)  # Handle NaN
                return np.nan_to_num(x, nan=0.0)  # Replace NaN with 0
            
            ts1_norm = robust_normalize(ts1)
            ts2_norm = robust_normalize(ts2)
            
            # Calculate DTW distance with proper warping window
            similarity = dtw(ts1_norm, ts2_norm, 
                           global_constraint="sakoe_chiba",
                           sakoe_chiba_radius=int(0.1*sample_size))
            
            component_similarities[comp][i,j] = similarity
            component_similarities[comp][j,i] = similarity  # Symmetric
    
    # Combine components with equal weighting
    combined_similarity = np.mean([component_similarities[comp] 
                                 for comp in ['HZ', 'HN', 'HE']], axis=0)
    
    # Adaptive thresholding - find natural breaks in similarity distribution
    if np.any(combined_similarity > 0):
        # Use log scale for threshold determination to handle skewed distributions
        log_sim = np.log1p(combined_similarity[combined_similarity > 0])
        sim_threshold = np.exp(np.percentile(log_sim, 10)) - 1
        ts_adjacency = (combined_similarity <= sim_threshold).astype(int)
    else:
        ts_adjacency = np.zeros_like(combined_similarity, dtype=int)
    
    np.fill_diagonal(ts_adjacency, 0)  # No self-similarity
    
    return ts_adjacency, combined_similarity, component_similarities

def calculate_stability(node_features, station_data):
    """Classify stations as stable/unstable with anomaly detection"""
    stability_scores = []
    anomaly_flags = []
    extreme_threshold = 0.95
    consistency_threshold = 0.9
    temporal_window = 24*60  # 24 hours in minutes
    
    for idx, (_, features) in enumerate(node_features.iterrows()):
        station = features['station']
        score = 0
        anomaly = False
        temporal_anomaly = False
        
        # Extreme value detection
        key_features = [
            'peak_amplitude_max_HZ', 'acceleration_max_HZ',
            'velocity_max_HZ', 'displacement_max_HZ',
            'rms_energy_max_HZ', 'dominant_freq_max_HZ'
        ]
        
        for feat in key_features:
            if feat in features and features[feat] > node_features[feat].quantile(extreme_threshold):
                score += 1
        
        # Feature consistency check
        if (features['velocity_max_HZ'] > node_features['velocity_max_HZ'].quantile(0.9) and
            features['displacement_max_HZ'] < node_features['displacement_max_HZ'].quantile(0.1)):
            score += 2
        
        # Component correlation and temporal analysis
        if station in station_data:
            components = station_data[station]
            try:
                # Component correlation
                recent_hz = components['HZ'][['acceleration', 'velocity', 'displacement']].iloc[-100:].mean()
                recent_hn = components['HN'][['acceleration', 'velocity', 'displacement']].iloc[-100:].mean()
                recent_he = components['HE'][['acceleration', 'velocity', 'displacement']].iloc[-100:].mean()
                
                hz_hn_corr = np.corrcoef(recent_hz, recent_hn)[0,1]
                hz_he_corr = np.corrcoef(recent_hz, recent_he)[0,1]
                hn_he_corr = np.corrcoef(recent_hn, recent_he)[0,1]
                
                if (hz_hn_corr < consistency_threshold or 
                    hz_he_corr < consistency_threshold or 
                    hn_he_corr < consistency_threshold):
                    anomaly = True
                    score += 3
                
                # Temporal anomaly detection
                for comp in ['HZ', 'HN', 'HE']:
                    df = components[comp]
                    if len(df) >= temporal_window:
                        df['rolling_std'] = df['acceleration'].rolling(temporal_window).std()
                        current_std = df['rolling_std'].iloc[-1]
                        median_std = df['rolling_std'].median()
                        if current_std > 3 * median_std:
                            temporal_anomaly = True
                            score += 2
            except Exception as e:
                print(f"Error processing {station}: {str(e)}")
                continue
        
        stability_scores.append(score)
        anomaly_flags.append(anomaly or temporal_anomaly)
    
    # Normalize and classify
    stability_scores = np.array(stability_scores)
    normalized_scores = (stability_scores - np.min(stability_scores)) / (np.max(stability_scores) - np.min(stability_scores))
    
    node_features['stability_score'] = normalized_scores
    node_features['anomaly'] = anomaly_flags
    
    conditions = [
        (node_features['anomaly'] == True),
        (node_features['stability_score'] > 0.7),
        (node_features['stability_score'] <= 0.7)
    ]
    choices = ['unstable', 'unstable', 'stable']
    node_features['stability'] = np.select(conditions, choices)
    
    return node_features

def build_graph(adjacency_matrix, node_features, similarity_matrix=None):
    """Construct network graph with node attributes"""
    G = nx.from_numpy_array(adjacency_matrix)
    
    for i, row in node_features.iterrows():
        G.nodes[i]['station'] = row['station']
        G.nodes[i]['stability'] = row['stability']
        G.nodes[i]['stability_score'] = row['stability_score']
        G.nodes[i]['lat'] = row['receiver_latitude']
        G.nodes[i]['lon'] = row['receiver_longitude']
        
        if similarity_matrix is not None:
            G.nodes[i]['similarity'] = similarity_matrix[i].tolist()
    
    # Calculate risk scores
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        unstable_neighbors = [n for n in neighbors if G.nodes[n]['stability'] == 'unstable']
        
        risk_score = len(unstable_neighbors) / len(neighbors) if neighbors else 0
        G.nodes[node]['risk_score'] = risk_score
        
        if G.nodes[node]['stability'] == 'stable' and risk_score > 0.3:
            G.nodes[node]['risk'] = 'high'
        elif G.nodes[node]['stability'] == 'stable' and risk_score > 0.1:
            G.nodes[node]['risk'] = 'medium'
        else:
            G.nodes[node]['risk'] = 'low'
    
    return G

def create_hybrid_network(geo_data, ts_data, geo_weight=0.5, ts_weight=0.5):
    """Creates hybrid network combining geological and time-series networks"""
    hybrid_nodes = geo_data['nodes'].copy()
    hybrid_links = []
    
    # Create dictionaries for quick lookup
    geo_links = {(link['source'], link['target']): 1 for link in geo_data['links']}
    ts_links = {(link['source'], link['target']): 1 for link in ts_data['links']}
    
    all_pairs = set(geo_links.keys()).union(set(ts_links.keys()))
    
    for source, target in all_pairs:
        geo_strength = geo_links.get((source, target), 0)
        ts_strength = ts_links.get((source, target), 0)
        weight = geo_weight * geo_strength + ts_weight * ts_strength
        
        if weight > 0:
            hybrid_links.append({
                'source': source,
                'target': target,
                'weight': weight,
                'geo_strength': geo_strength,
                'ts_strength': ts_strength
            })
    
    return {
        'nodes': hybrid_nodes,
        'links': hybrid_links
    }

def prepare_visualization_data(G, similarity_matrix=None, station_data_folder=None):
    """Convert graph to visualization-ready format and embed time-series"""
    node_colors = []
    for node in G.nodes():
        if G.nodes[node]['stability'] == 'unstable':
            node_colors.append('#ff0000')
        elif G.nodes[node]['risk'] == 'high':
            node_colors.append('#ff9900')
        elif G.nodes[node]['risk'] == 'medium':
            node_colors.append('#ffcc00')
        else:
            node_colors.append('#00aa00')

    nodes = []
    for i, node in enumerate(G.nodes()):
        station_id = G.nodes[node]['station']
        node_data = {
            'id': station_id,
            'latitude': G.nodes[node]['lat'],
            'longitude': G.nodes[node]['lon'],
            'stability': G.nodes[node]['stability'],
            'stability_score': float(G.nodes[node]['stability_score']),
            'risk': G.nodes[node]['risk'],
            'risk_score': float(G.nodes[node]['risk_score']),
            'color': node_colors[i]
        }

        if similarity_matrix is not None:
            node_data['similarity'] = float(similarity_matrix[node, node])

        # Embed time-series with precision preservation
        time_series = {}
        if station_data_folder:
            station_path = os.path.join(station_data_folder, station_id)
            for comp in ['HZ', 'HN', 'HE']:
                file_path = os.path.join(station_path, f"{station_id}_{comp}.csv")
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path).iloc[::10]  # Subsampling only
                    
                    # Preserve small values without unnecessary rounding
                    time_series[comp] = {
                        'acceleration': [float(x) for x in df['acceleration'].values],
                        'velocity': [float(x) for x in df['velocity'].values],
                        'displacement': [float(x) for x in df['displacement'].values]
                    }
                    
        node_data['time_series'] = time_series
        nodes.append(node_data)

    links = []
    for edge in G.edges():
        link_data = {
            'source': G.nodes[edge[0]]['station'],
            'target': G.nodes[edge[1]]['station'],
            'value': 1
        }
        if similarity_matrix is not None:
            link_data['similarity'] = float(similarity_matrix[edge[0], edge[1]])
        links.append(link_data)

    return {'nodes': nodes, 'links': links}


def visualize_three_networks(data):
    """Visualizes geological, similarity, and hybrid networks side-by-side"""
    # Create hybrid network if not exists
    if 'hybrid_network' not in data:
        data['hybrid_network'] = create_hybrid_network(
            data['geological_network'],
            data['similarity_network']
        )
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Geological Network', 'Similarity Network', 'Hybrid Network'),
        specs=[[{'type': 'geo'}, {'type': 'geo'}, {'type': 'geo'}]]
    )
    
    # Visualization settings
    edge_color = '#888'
    node_size = 8
    
    # Add all three networks to subplots
    for col, network_type in enumerate(['geological_network', 'similarity_network', 'hybrid_network'], 1):
        network_data = data[network_type]
        
        # Add edges
        for link in network_data['links']:
            source_node = next(node for node in network_data['nodes'] if node['id'] == link['source'])
            target_node = next(node for node in network_data['nodes'] if node['id'] == link['target'])
            
            fig.add_trace(
                go.Scattergeo(
                    lon=[source_node['longitude'], target_node['longitude']],
                    lat=[source_node['latitude'], target_node['latitude']],
                    mode='lines',
                    line=dict(width=0.5, color=edge_color),
                    hoverinfo='none',
                    showlegend=False
                ),
                row=1, col=col
            )
        
        # Add nodes
        node_trace = go.Scattergeo(
            lon=[node['longitude'] for node in network_data['nodes']],
            lat=[node['latitude'] for node in network_data['nodes']],
            text=[f"{node['id']}<br>Stability: {node['stability']}<br>Risk: {node['risk']}" 
                  for node in network_data['nodes']],
            marker=dict(
                size=node_size,
                color=[node['color'] for node in network_data['nodes']],
                line_width=0.5
            ),
            hoverinfo='text',
            showlegend=False
        )
        fig.add_trace(node_trace, row=1, col=col)
    
    # Update layout
    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="rgb(243, 243, 243)",
        countrycolor="rgb(204, 204, 204)",
        showcountries=True
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        title_text="Seismic Network Analysis",
        title_x=0.5
    )
    
    # Add color legend
    colors = ['red', 'orange', 'yellow', 'green']
    labels = ['Unstable', 'High Risk', 'Medium Risk', 'Stable']
    
    for color, label in zip(colors, labels):
        fig.add_trace(
            go.Scattergeo(
                lon=[None],
                lat=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=label,
                hoverinfo='none'
            )
        )
    
    fig.show()
    return data

def main():
    # Configuration
    NODE_FEATURES_PATH = r"C:\Users\medma\Desktop\seismic-visualizer\datasets\node_features_merged.csv"
    STATION_DATA_FOLDER = r"C:\Users\medma\Desktop\seismic-visualizer\datasets\station_csv_output"
    OUTPUT_FILE = r"C:\Users\medma\Desktop\seismic-visualizer\datasets\seismic_network.json"
    
    print("Loading data...")
    node_features, station_data = load_all_data(NODE_FEATURES_PATH, STATION_DATA_FOLDER)
    
    print("Calculating geographical similarity...")
    geo_adjacency, geo_distances = calculate_geo_similarity(node_features)
    
    print("Calculating time-series similarity...")
    ts_adjacency, ts_similarity, component_similarities = calculate_ts_similarity(
        station_data, node_features
    )
    
    print("Analyzing stability with anomaly detection...")
    node_features = calculate_stability(node_features, station_data)
    
    print("Building graphs...")
    geo_graph = build_graph(geo_adjacency, node_features)
    ts_graph = build_graph(ts_adjacency, node_features, ts_similarity)
    
    print("Preparing visualization data...")
    geo_viz_data = prepare_visualization_data(geo_graph, station_data_folder=STATION_DATA_FOLDER)
    ts_viz_data = prepare_visualization_data(ts_graph, ts_similarity, station_data_folder=STATION_DATA_FOLDER)

    print("Creating hybrid network...")
    hybrid_data = create_hybrid_network(geo_viz_data, ts_viz_data)
    
    # Prepare output data
    output_data = {
        'geological_network': geo_viz_data,
        'similarity_network': ts_viz_data,
        'hybrid_network': hybrid_data,
        'component_similarities': {
            'HZ': component_similarities['HZ'].tolist(),
            'HN': component_similarities['HN'].tolist(),
            'HE': component_similarities['HE'].tolist()
        },
        'geo_distances': geo_distances.tolist(),
        'color_legend': {
            '#ff0000': 'Unstable (red)',
            '#ff9900': 'High Risk (orange)',
            '#ffcc00': 'Medium Risk (yellow)',
            '#00aa00': 'Stable (green)'
        }
    }
    
    # Save output
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f)
    
    print(f"Processing complete. Output saved to {OUTPUT_FILE}")
    
    # Visualize all three networks
    print("Generating visualizations...")
    visualize_three_networks(output_data)

if __name__ == "__main__":
    main()