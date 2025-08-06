
import json
import time
from kafka import KafkaProducer

# Kafka Producer setup
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Load seismic network data
with open("./data/seismic_network.json", "r") as f:
    data = json.load(f)
    data = data.get("hybrid_network")


components = ['HZ', 'HN', 'HE']

# Determine shortest time series length for safety
min_len = min(
    len(node['time_series'][comp]['acceleration'])
    for node in data['nodes']
    for comp in components
    if comp in node['time_series']
)

# Stream raw values at each time step for all stations and components
for step in range(min_len):
    message = {
        "step": step,
        "timestamp": time.time(),
        "data": []
    }

    for node in data['nodes']:
        station_id = node['id']

        for comp in components:
            ts = node['time_series'].get(comp)
            if not ts:
                continue

            message["data"].append({
                "station_id": station_id,
                "component": comp,
                "acceleration": ts['acceleration'][step],
                "velocity": ts['velocity'][step],
                "displacement": ts['displacement'][step]
            })

    # Send one full frame for this time step
    producer.send("seismic_data", value=message)
    print(f"Sent: step [{step}]") 
    time.sleep(1)  # Simulate 1Hz streaming