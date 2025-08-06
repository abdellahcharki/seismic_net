from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "seismic_data",
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

for msg in consumer:
    print("Received:", msg.value)
