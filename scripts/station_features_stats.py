import os
import pandas as pd
import numpy as np
import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from scipy.stats import skew, kurtosis
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph

# === SETTINGS ===
DATA_DIR = r"C:\Users\medma\Desktop\seismic-visualizer\datasets\waveformsF"  # Folder containing PAH, PAL, etc.
CHANNELS = ["HHN", "HHE", "HHZ"]
QUANTITIES = ["acceleration", "velocity", "displacement"]
STATS_PER_GROUP = 6

# === 1. Get all station folders ===
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Directory not found: {DATA_DIR}")

stations = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

# === 2. Define statistical feature extractor ===
def extract_stats(values):
    return [
        np.mean(values),
        np.std(values),
        np.max(values),
        np.min(values),
        skew(values),
        kurtosis(values)
    ]

# === 3. Load station data and compute features ===
def load_station_statistics_all(station_name):
    station_path = os.path.join(DATA_DIR, station_name)
    files = [f for f in os.listdir(station_path) if f.endswith('.csv')]

    final_features = []

    for quantity in QUANTITIES:
        channel_data = {ch: [] for ch in CHANNELS}

        for file in files:
            file_path = os.path.join(station_path, file)
            try:
                df = pd.read_csv(file_path)
                matched_channel = next((ch for ch in CHANNELS if ch in file), None)
                if matched_channel and quantity in df.columns:
                    values = df[quantity].dropna().values
                    channel_data[matched_channel].append(values)
            except Exception as e:
                print(f"Warning: could not read {file_path}: {e}")

        for ch in CHANNELS:
            if channel_data[ch]:
                merged = np.concatenate(channel_data[ch])
                stats = extract_stats(merged)
            else:
                stats = [0.0] * STATS_PER_GROUP
            final_features.extend(stats)

    return torch.tensor(final_features, dtype=torch.float32)

# Prepare data to save: one row per station with all channel stats concatenated
rows = []
for station in stations:
    features_tensor = G.nodes[station]["x"]
    features_list = features_tensor.tolist()
    row = [station] + features_list
    rows.append(row)

# Build column names
# For example: for each quantity and channel, you have 6 stats: mean, std, max, min, skew, kurtosis
# Assuming order: for each quantity, for each channel, 6 stats
stat_names = ['mean', 'std', 'max', 'min', 'skew', 'kurtosis']
columns = ['station']

for quantity in QUANTITIES:
    for ch in CHANNELS:
        for stat in stat_names:
            columns.append(f"{quantity}_{ch}_{stat}")

# Create DataFrame and save
df_features = pd.DataFrame(rows, columns=columns)

df_features.to_csv("station_features_stats.csv", index=False)

print("✅ Station features saved to station_features_stats.csv")