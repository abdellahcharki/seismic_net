
import os
import pandas as pd
from glob import glob
import numpy as np

# Load metadata
merged_df = pd.read_csv(r"C:\Users\medma\Desktop\seismic-visualizer\datasets\merged.csv")

# Directory where waveform CSVs are stored
waveform_dir = r"C:\Users\medma\Desktop\seismic-visualizer\datasets\waveformsF"
all_data = []

# Process each waveform file
for station_folder in os.listdir(waveform_dir):
    station_path = os.path.join(waveform_dir, station_folder)
    if not os.path.isdir(station_path):
        continue
    
    for waveform_file in glob(os.path.join(station_path, '*.csv')):
        # Extract event_id and channel from filename
        filename = os.path.basename(waveform_file)
        parts = filename.split('_')
        event_id = '_'.join(parts[:3])  # PAH.NN_20181228212814_EV
        channel = parts[3]              # HHN, HHE, HHZ
        
        # Load waveform data
        waveform_data = pd.read_csv(waveform_file)
        
        # Add identifier columns
        waveform_data['station'] = parts[0].split('.')[0]  # PAH
        waveform_data['event_id'] = event_id
        waveform_data['channel'] = channel
        
        # Merge with metadata
        metadata = merged_df[
            (merged_df['event_id'] == event_id) & 
            (merged_df['channel'] == channel)
        ]
        
        if not metadata.empty:
            # Add metadata columns to waveform data
            metadata = metadata.drop(columns=['event_id', 'channel']).iloc[0]
            for col, value in metadata.items():
                waveform_data[col] = value
            
            all_data.append(waveform_data)

# Combine all data into one DataFrame
combined_df = pd.concat(all_data, ignore_index=True)

# Save to single CSV
combined_df.to_csv('combined_waveform_metadata.csv', index=False)
print("All data merged into single CSV!")





# === Load the dataset ===
df = pd.read_csv(r"C:\Users\medma\Desktop\seismic-visualizer\datasets\combined_waveform_metadata.csv")

# === Drop rows with missing 'station' or 'channel' ===
df = df.dropna(subset=["station", "channel"])

# === List of features to aggregate ===
features = [
    "acceleration", "velocity", "displacement",
    "peak_amplitude", "rms_energy", "zero_crossings", "duration",
    "spectral_centroid", "spectral_bandwidth", "dominant_freq", "snr_db"
]

# === Convert feature columns to numeric (force errors to NaN) ===
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# === Group and aggregate ===
agg_funcs = ['mean', 'std', 'min', 'max']
df_stats = df.groupby(['station', 'channel'])[features].agg(agg_funcs)

# === Flatten column names ===
df_stats.columns = ['{}_{}'.format(f, stat) for f, stat in df_stats.columns]
df_stats.reset_index(inplace=True)

# === Pivot to wide format (channels as suffixes) ===
df_pivot = df_stats.pivot(index='station', columns='channel')

# Flatten column multi-index: (acceleration_mean, HHN) → acceleration_mean_HHN
df_pivot.columns = ['{}_{}'.format(stat, ch) for stat, ch in df_pivot.columns]
df_pivot.reset_index(inplace=True)

# === Fill any missing values with 0 (e.g., if a station lacks a channel) ===
df_pivot.fillna(0, inplace=True)
# 1. Rename columns by removing the character before the last two letters if exists
def rename_col(col):
    # If length is at least 3 and third last char is an extra one, remove it
    if len(col) > 3:
        # For example, 'acceleration_mean_BHE'
        # Remove the character before last two letters: index -3
        # Result: 'acceleration_mean_HE'
        return col[:-3] + col[-2:]
    else:
        return col

# Apply renaming only for columns except 'station'
new_columns = {}
for col in df_pivot.columns:
    if col != 'station':
        new_columns[col] = rename_col(col)
    else:
        new_columns[col] = col

# Rename the DataFrame columns
df_renamed = df_pivot.rename(columns=new_columns)
# Step 1: Keep 'station' column as is (if you want to preserve it)
station_col = 'station'  # replace with your station column name if different

# Step 2: Select all other columns (the ones to merge by name)
data_cols = [col for col in df_renamed.columns if col != station_col]
# Keep 'station' column as is
result_df = pd.DataFrame()
result_df['station'] = df_renamed['station']

# Get unique column names excluding 'station'
unique_cols = [col for col in df_renamed.columns if col != 'station']
unique_cols = list(set(unique_cols))  # unique names

for col_name in unique_cols:
    # Select all duplicate columns with this name
    duplicate_cols = [c for c in df_renamed.columns if c == col_name]
    
    # Subset of these duplicate columns
    data_subset = df_renamed[duplicate_cols]
    
    # Replace zeros with NaN so zeros don't affect the mean
    data_subset_no_zero = data_subset.replace(0, np.nan)
    
    # Take the mean across duplicates ignoring zeros (NaN)
    # If multiple non-zero values exist, average them
    # If all zeros, result will be NaN -> fill back with zero
    merged_col = data_subset_no_zero.mean(axis=1).fillna(0)
    
    # Add merged column to result dataframe
    result_df[col_name] = merged_col


result_df = result_df.loc[:, ~(df == 0).all()]

merged_selected = merged_df[['station', 'receiver_latitude', 'receiver_longitude', 'receiver_elevation_m','source_depth_km']]
merged_unique = merged_selected.drop_duplicates(subset='station')
result_df = result_df.merge(merged_unique, on='station', how='left')

# === Save result ===
output_path = r"C:\Users\medma\Desktop\seismic-visualizer\datasets\node_features_merged.csv"
result_df.to_csv(output_path, index=False)

print(f"✅ node_features_merged.csv created successfully. Shape: {result_df.shape}")