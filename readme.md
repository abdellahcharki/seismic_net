# 🌍 Seismic Network Real-Time Monitoring and Analysis Tool

This project provides a real-time platform for monitoring seismic stations, visualizing their interconnections, and detecting potential risks using a Graph Neural Network (GNN) model.

## 🚀 Features

- 📈 Real-time time-series data ingestion via **Kafka**
- 🧠 GCN model predicts station stability from spatio-temporal features
- 🗺️ Frontend map and graph visualization with **React + D3 Force Graph**
- 📊 Live time-series charts using **Recharts**
- 🧩 Graph and station data stored in **Neo4j** and **TimescaleDB**
- 🔌 Bi-directional updates using **Flask-SocketIO**

## 🗂 Project Structure

```
seismic_net/
├── backend/                                # Backend for processing, classification, and API
│   ├── app.py                              # Main Flask app serving REST + Socket.IO
│   ├── stream_consumer.py                  # Kafka consumer to handle real-time classification
│   ├── Dockerfile.backend                  # Docker setup for the backend container
│   ├── requirements.txt                    # Backend Python dependencies
│   ├── models/
│   │   ├── gcn.py                          # GCN model definition using PyTorch
│   │   └── gnn_model.pt                    # Trained GNN model weights
│   └── utils/
│       └── graph_utils.py                  # Graph-based helper functions (e.g., similarity, building edges)
│
├── data/                                   # Input data folder
│   └── seismic_network.json                # Seismic station network: metadata + time series
│
├── frontend/                               # React frontend for map, graph, and time series UI
│   ├── public/
│   │   ├── ....                     
│   │   └── index.html                      # Main HTML template where React mounts the app
│   ├── src/
│   │   ├── App.jsx                         # Root React component
│   │   ├── index.js                        # Entry point rendering the app to DOM
│   │   ├── logo.png                        # Logo used in the frontend (header or splash)
│   │   ├── Components/
│   │   │   ├── Graph.jsx                   # Graph visualization (e.g., nodes + links)
│   │   │   ├── Map.jsx                     # Leaflet map with sensor markers
│   │   │   ├── TimeSeries.jsx              # Plotly time-series viewer for seismic data
│   │   │   ├── StationDetails.jsx          # Info box for selected station (risk, stability, etc.)
│   │   │   ├── StationFilter.jsx           # UI for filtering stations (by status, component, etc.)
│   │   │   ├── InfoSide.jsx                # Sidebar with metadata and network summary
│   │   │   ├── NavBar.jsx                  # Navigation bar across views
│   │   │   ├── Loading.jsx                 # Spinner or loading indicator
│   │   │   └── service.js                  # Axios-based API calls to backend (stations, links)
│   │   ├── img/                            # Folder for image assets (if used)
│   │   └── scss/                           # SCSS stylesheets for component styling
│   ├── .gitignore                          # Files to ignore in Git versioning
│   ├── package.json                        # Project metadata and NPM dependencies
│   ├── package-lock.json                   # Exact dependency versions for reproducibility
│   └── README.md                           # Instructions/documentation for frontend
│
├── scripts/                                # Notebooks and Python scripts for model building and analysis
│   ├── Build_node_features.py              # Extract node features (for GNN input)
│   ├── Build_station_csv_output.py         # Save processed results to CSV (e.g., station-level summaries)
│   ├── gnn_model.pt                        # Copy of trained model (for testing/inference)
│   ├── gnn_train.ipynb                     # Jupyter notebook for training GNN model
│   ├── metadata_merge.py                   # Combine different sources of station metadata
│   ├── processor.py                        # Preprocess time series and metadata
│   ├── seismic_network_analyzer.py         # Analyze network structure, compute centrality/stability
│   └── station_features_stats.py           # Generate statistics for station features
│
├── docker-compose.yml                      # Orchestrates Kafka, backend, frontend, and database services
├── producer.py                             # Streams seismic time-series data to Kafka topic
├── readme.md                               # Main project documentation (setup, usage, architecture)
├── run.cmd                                 # Windows batch script to start backend/frontend
└── run_stream.cmd                          # Script to start Kafka data producer

```

## ⚙️ Setup Instructions

### 🐳 Using Docker

1. Start infrastructure:
   ```bash
   docker-compose up -d
   ```

2. Build and run backend:
   ```bash
   cd backend
   docker build -t flask-backend -f ../Dockerfile.backend .
   docker run -p 5000:5000 --network=host flask-backend
   ```

### 🖥️ Run Locally

1. **Backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   python app.py
   ```

2. **Frontend**
   ```bash
   cd frontend
   npm install
   npm start
   ```

3. **Kafka Producer**
   ```bash
   cd kafka
   python producer.py
   ```

4. (Optional) Use `run.cmd` to launch everything on Windows.

4. (Optional) Use `run_stream` to startthe stream Kafka.

## 📡 Kafka Topic

| Topic Name     | Description                     |
|----------------|----------------------------------|
| `seismic_data` | Real-time incoming station data |

## 🧠 GNN Model

- **Model Type**: Graph Convolutional Network (GCN)
- **Input**: Node features based on last 10 seconds of signal summary
- **Output**: Station state classification: `stable` or `unstable`
- **Inference**: Happens on every Kafka message
- **Training**: Performed offline. Model saved at `models/gnn_model.pt`

## 📊 Frontend Visualization

- **Node Colors**:
  - 🔴 Red: Unstable station
  - 🟠 Orange: Stable with high risk
  - 🟢 Green: Stable with low risk
  - ⚪ Gray: Unknown

- **WebSocket Updates**:
  - `seismic_update`: triggers time-series chart updates
  - `station_update`: triggers station color/status updates

## 🧪 Testing Real-Time Updates

To simulate live data:
```bash
python producer.py
```

This will stream new station readings every second, updating Neo4j, TimescaleDB, and the frontend UI.

## 📚 Technologies Used

- **Backend**: Python, Flask, Flask-SocketIO, Kafka, Neo4j, TimescaleDB, PyTorch Geometric
- **Frontend**: React, Recharts, react-force-graph, Socket.IO
- **Data Storage**: TimescaleDB (signal data), Neo4j (station metadata)

## 📬 Project Info

Developed by **Abdellah Charki** and **Mohamed El Maghari**  
Supervised by **Mouna Ammar**  
Master’s in Data Science – Leipzig University, SoSe 2025