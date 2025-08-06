from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import threading
from kafka import KafkaConsumer
from neo4j import GraphDatabase
import psycopg2
from datetime import datetime
import torch
from models.gcn import MyGCN
from utils.graph_utils import extract_features_and_graph
import json

# Flask + Config
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Database connections
NEO4J_URI = "bolt://localhost:7687"
DB_USER = "neo4j"
DB_PASSWORD = "strongpass123"
driver = GraphDatabase.driver(NEO4J_URI, auth=(DB_USER, DB_PASSWORD))
conn = psycopg2.connect(dbname="seismicdb", user=DB_USER, password=DB_PASSWORD)

# Load GCN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyGCN(input_dim=2, hidden_dim=16, output_dim=2)
model.load_state_dict(torch.load("models/gnn_model.pt", map_location=device))
model.eval()

# Insert timeseries into TimescaleDB
def insert_timeseries(station_id, component, signal_data, timestamp):
    cursor = conn.cursor()
    for signal_type, value in signal_data.items():
        cursor.execute(
            "INSERT INTO station_timeseries (station_id, sensor_component, signal_type, value, timestamp) VALUES (%s, %s, %s, %s, %s)",
            (station_id, component, signal_type, value, timestamp)
        )
    conn.commit()
    cursor.close()

# Kafka consumer thread
def consume_kafka():
    consumer = KafkaConsumer(
        'seismic_data',
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest'
    )

    prev_status = {}

    for msg in consumer:
        payload = msg.value
        timestamp = datetime.now()

        for record in payload.get("data", []):
            insert_timeseries(record["station_id"], record["component"], {
                "acceleration": record["acceleration"],
                "velocity": record["velocity"],
                "displacement": record["displacement"]
            }, timestamp)

        x, edge_index, station_ids = extract_features_and_graph(conn, driver)
        with torch.no_grad():
            predictions = torch.argmax(model(x, edge_index), dim=1).tolist()

        for idx, sid in enumerate(station_ids):
            status = "unstable" if predictions[idx] == 1 else "stable"
            if prev_status.get(sid) != status:
                prev_status[sid] = status
                with driver.session() as session:
                    session.run("MATCH (s:Station {id: $id}) SET s.stability = $status",
                                id=sid, status=status)
                socketio.emit("station_update", {"station_id": sid, "status": status})

        socketio.emit("seismic_update", payload)

threading.Thread(target=consume_kafka, daemon=True).start()

# REST routes
@app.route('/hybrid/nodes')
def get_nodes():
    with driver.session() as session:
        result = session.run("MATCH (s:Station) RETURN s")
        return jsonify([dict(r['s']) for r in result])

@app.route('/hybrid/links')
def get_links():
    with driver.session() as session:
        result = session.run("MATCH (a)-[r:CONNECTED_TO]->(b) RETURN a.id AS source, b.id AS target")
        return jsonify([dict(r) for r in result])

@app.route('/hybrid/station/<station_id>')
def get_station(station_id):
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, sensor_component, signal_type, value FROM station_timeseries WHERE station_id = %s ORDER BY timestamp", (station_id,))
    rows = cursor.fetchall()
    cursor.close()

    timeseries = {"HE": {}, "HN": {}, "HZ": {}}
    for timestamp, comp, signal, value in rows:
        timeseries.setdefault(comp, {}).setdefault(signal, []).append(value)

    return jsonify({"id": station_id, "time_series": timeseries})

if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000)
