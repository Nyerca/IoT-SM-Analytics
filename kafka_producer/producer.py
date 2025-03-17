from kafka import KafkaProducer
import json
import time
import random
import time
from kafka.errors import KafkaError

print("Waiting...")

def create_producer():
    try:
        producer = KafkaProducer(
            bootstrap_servers="kafka:9092",
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        print("Connected to Kafka successfully!")
        return producer
    except KafkaError as e:
        print(f"Error connecting to Kafka: {e}")
        return None

# Try to create the consumer with retries
producer = None
while not producer:
    producer = create_producer()
    if not producer:
        print("Retrying to connect to Kafka...")
        time.sleep(5)  # Wait for 5 seconds before retrying



def generate_sensor_data():
    return {
        "sensorId": f"sensor_{random.randint(1, 5)}",
        "temperature": round(random.uniform(20, 100), 2),
        "timestamp": time.time(),
    }

while True:
    data = generate_sensor_data()
    producer.send("my-kafka-topic", value=data)
    print(f"Sent: {data}")
    time.sleep(2)
