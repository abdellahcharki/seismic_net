import os
import h5py
import pandas as pd
from tqdm import tqdm
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from data_loader import make_stream
from feature_extractor import extract_features_from_trace

def process_hdf5(hdf5_path, output_csv="features.csv", timeseries_dir="waveformsF"):
    """
    Main processing pipeline for reading HDF5, extracting waveform features,
    saving individual timeseries and a summary CSV.
    """
    client = Client("IRIS")
    dtfl = h5py.File(hdf5_path, 'r')
    results = []
    station_cache = {}

    os.makedirs(timeseries_dir, exist_ok=True)

    for key in tqdm(dtfl['data'].keys(), desc="Processing"):
        try:
            dataset = dtfl['data'][key]
            attrs = dataset.attrs
            station = attrs['receiver_code']
            network = attrs['network_code']
            trace_time = UTCDateTime(attrs['trace_start_time'])

            cache_key = f"{network}.{station}"
            if cache_key not in station_cache:
                try:
                    inventory = client.get_stations(
                        network=network, station=station,
                        starttime=trace_time,
                        endtime=trace_time + 60,
                        level="response", loc="*", channel="*"
                    )
                    station_cache[cache_key] = inventory
                except Exception as e:
                    print(f"⚠️ No response info for {cache_key}: {e}")
                    continue
            else:
                inventory = station_cache[cache_key]

            st_raw = make_stream(dataset)
            st_disp = st_raw.copy().remove_response(inventory=inventory, output="DISP", plot=False)
            st_vel  = st_raw.copy().remove_response(inventory=inventory, output="VEL", plot=False)
            st_acc  = st_raw.copy().remove_response(inventory=inventory, output="ACC", plot=False)

            for tr_d, tr_v, tr_a in zip(st_disp, st_vel, st_acc):
                if not (len(tr_d.data) == len(tr_v.data) == len(tr_a.data)):
                    print(f"⚠️ Length mismatch in {key}")
                    continue

                feat = extract_features_from_trace(tr_v)
                feat.update({
                    "event_id": key,
                    "station": tr_v.stats.station,
                    "channel": tr_v.stats.channel
                })
                results.append(feat)

                df = pd.DataFrame({
                    "time": tr_v.times(),
                    "acceleration": tr_a.data,
                    "velocity": tr_v.data,
                    "displacement": tr_d.data
                })

                station_dir = os.path.join(timeseries_dir, tr_v.stats.station)
                os.makedirs(station_dir, exist_ok=True)
                ts_filename = f"{key}_{tr_v.stats.channel}_timeseries.csv"
                df.to_csv(os.path.join(station_dir, ts_filename), index=False)

        except Exception as e:
            print(f"⚠️ Failed on {key}: {e}")

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\n✅ Features saved to {output_csv}")