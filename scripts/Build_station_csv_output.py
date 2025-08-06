import os
import pandas as pd
import numpy as np

def extract_component(filename):
    # Extract component from filename (last 2 characters before '_timeseries')
    base = os.path.basename(filename)
    match = base.split('_')[-2]
    return match[-2:]  # HE, HN, HZ

def load_full_station_timeseries(waveform_root):
    station_series = {}

    for station_folder in os.listdir(waveform_root):
        station_path = os.path.join(waveform_root, station_folder)
        if not os.path.isdir(station_path):
            continue

        component_data = {
            'HE': {'acceleration': [], 'velocity': [], 'displacement': []},
            'HN': {'acceleration': [], 'velocity': [], 'displacement': []},
            'HZ': {'acceleration': [], 'velocity': [], 'displacement': []},
        }

        for file in os.listdir(station_path):
            if not file.endswith('.csv'):
                continue

            filepath = os.path.join(station_path, file)
            component = extract_component(file)
            if component not in component_data:
                continue

            df = pd.read_csv(filepath)
            for feature in ['acceleration', 'velocity', 'displacement']:
                if feature in df.columns:
                    series = df[feature].reset_index(drop=True)
                    component_data[component][feature].append(series)

        # Average across all traces per component + feature
        averaged_data = {}
        for comp, features in component_data.items():
            averaged_data[comp] = {}
            for feature, series_list in features.items():
                if not series_list:
                    continue
                max_len = max(len(s) for s in series_list)
                padded = [s.reindex(range(max_len), fill_value=np.nan) for s in series_list]
                combined_df = pd.concat(padded, axis=1)
                averaged = combined_df.mean(axis=1, skipna=True)
                averaged_data[comp][feature] = averaged

        station_series[station_folder] = averaged_data

    return station_series


import os
import pandas as pd

def save_station_series_to_csv(station_series, output_dir="station_csv_output"):
    os.makedirs(output_dir, exist_ok=True)

    for station, components in station_series.items():
        station_dir = os.path.join(output_dir, station)
        os.makedirs(station_dir, exist_ok=True)

        for comp, signals in components.items():
            df = pd.DataFrame(signals)  # keys: acceleration, velocity, displacement
            filename = f"{station}_{comp}.csv"
            df.to_csv(os.path.join(station_dir, filename), index=False)

    print(f"All station time series saved to {output_dir}/")

# Usage
station_series = load_full_station_timeseries(r"C:\Users\medma\Desktop\seismic-visualizer\datasets\waveformsF") 
save_station_series_to_csv(station_series)