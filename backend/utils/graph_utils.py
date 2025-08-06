def extract_features_and_graph(conn, driver):
    import torch
    from torch_geometric.utils import from_networkx
    import networkx as nx

    # Step 1: Load stations from Neo4j
    with driver.session() as session:
        nodes = session.run("MATCH (s:Station) RETURN s.id AS id, s.risk AS risk, s.stability AS stability").data()
        edges = session.run("""
            MATCH (s1:Station)-[:CONNECTED_TO]->(s2:Station)
            RETURN s1.id AS source, s2.id AS target
        """).data()

    station_ids = [n["id"] for n in nodes]
    id_index = {id_: i for i, id_ in enumerate(station_ids)}
    num_stations = len(station_ids)

    # Step 2: Build edge_index
    edge_index = []
    for e in edges:
        src = id_index.get(e["source"])
        tgt = id_index.get(e["target"])
        if src is not None and tgt is not None:
            edge_index.append((src, tgt))

    # Step 3: Compute 2 input features
    # Example: 1. average velocity from TimescaleDB, 2. number of unstable neighbors (or dummy)
    x = torch.zeros((num_stations, 2))

    # Fetch average velocity per station
    cursor = conn.cursor()
    for i, sid in enumerate(station_ids):
        cursor.execute("""
            SELECT AVG(value) FROM station_timeseries
            WHERE station_id = %s AND signal_type = 'velocity'
        """, (sid,))
        avg_vel = cursor.fetchone()[0]
        x[i][0] = avg_vel if avg_vel else 0.0

    # Count # of unstable neighbors from Neo4j
    for i, sid in enumerate(station_ids):
        q = """
        MATCH (s:Station {id: $id})-[:CONNECTED_TO]-(n:Station)
        RETURN count(CASE WHEN n.stability = 'unstable' THEN 1 END) AS cnt
        """
        result = driver.session().run(q, id=sid).single()
        x[i][1] = result["cnt"] if result else 0

    cursor.close()

    # Convert edge_index to tensor
    import torch_geometric
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return x, edge_index, station_ids