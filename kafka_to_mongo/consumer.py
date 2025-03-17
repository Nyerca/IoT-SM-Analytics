from kafka import KafkaConsumer
import json
from pymongo import MongoClient




from pymongo.errors import ServerSelectionTimeoutError
from kafka.errors import KafkaError
import json
import time








# Retry logic to handle NoBrokersAvailable error
def create_consumer():
    try:
        consumer = KafkaConsumer(
            "my-kafka-topic",
            bootstrap_servers="kafka:9092",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        print("Connected to Kafka successfully!")
        return consumer
    except KafkaError as e:
        print(f"Error connecting to Kafka: {e}")
        return None

# Try to create the consumer with retries
consumer = None
while not consumer:
    consumer = create_consumer()
    if not consumer:
        print("Retrying to connect to Kafka...")
        time.sleep(5)  # Wait for 5 seconds before retrying

print("Listening for messages from Kafka...")


import time

time.sleep(10)  # Wait for MongoDB replica set initialization

client = MongoClient("mongodb://mongo1:27017,mongo2:27017,mongo3:27017/?replicaSet=rs0")

db = client.iot_db
collection = db.sensor_data


for message in consumer:
    sensor_data = message.value
    print(f"Storing to MongoDB: {sensor_data}")
    collection.insert_one(sensor_data)
    print(f"Saved: {sensor_data}")

