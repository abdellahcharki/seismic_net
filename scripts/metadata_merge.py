import pandas as pd

def filter_and_save_metadata(input_csv, output_csv):
    """
    Filters metadata for local earthquakes with magnitude > 3 and distance <= 20 km.
    """
    df = pd.read_csv(input_csv)
    df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km <= 20) & (df.source_magnitude > 3)]
    df.to_csv(output_csv, index=False)


def merge_metadata_with_features(meta_csv, feature_csv, output_csv="merged.csv"):
    """
    Merge cleaned metadata with extracted features based on station and event ID.
    """
    meta = pd.read_csv(meta_csv)
    features = pd.read_csv(feature_csv)

    meta = meta.rename(columns={'receiver_code': 'station', 'trace_name': 'event_id'})
    merged = pd.merge(features, meta, on=['station', 'event_id'], how='inner')
    merged.to_csv(output_csv, index=False)
    return merged
