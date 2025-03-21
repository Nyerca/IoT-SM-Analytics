from kafka import KafkaConsumer
import json
from pymongo import MongoClient

from pymongo.errors import ServerSelectionTimeoutError
from kafka.errors import KafkaError
import json
import time
import pandas as pd

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

def find_timestamp_n_days(current_time, days:30):
    seconds_in_n_days = days * 86400
    return current_time - seconds_in_n_days

def get_from_db(machine_id, timestamp):
    print("**get machine id: " + str(machine_id))

    # Query MongoDB for all records matching the given machine_id
    cursor = collection.find({"machine_id": machine_id, "timestamp": {"$gte": find_timestamp_n_days(timestamp, 30)} })

    # Convert cursor to a list to count the documents
    documents = list(cursor)
    print("Number of documents found:", len(documents))

    return documents


def to_dataframe(sensor_data, old_documents):
    # Add the sensor_data to the list of old_documents
    all_data = old_documents + [sensor_data]

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(all_data)

    # Sort the DataFrame by the 'timestamp' column in ascending order
    df_sorted = df.sort_values(by='timestamp', ascending=True).reset_index(drop=True)

    return df_sorted


def set_days_since_last_failure(df):
    # Compute "Time since last failure" (days since last label=1 per machine) Starting from 1
    df['days_since_last_failure'] = 1

    # Iterate over the DataFrame to calculate the days since the last failure
    for i in range(1, len(df)):
        # Check if the current row's machine_id is the same as the previous one and if the date has changed
        #current_date = df.loc[i, 'timestamp'].date()
        #prev_date = df.loc[i-1, 'timestamp'].date()

        # Convert timestamps to datetime objects and extract the YEAR-MONTH-DAY
        current_date = datetime.fromtimestamp(df.loc[i, 'timestamp']).strftime('%Y-%m-%d')
        prev_date= datetime.fromtimestamp(df.loc[i-1, 'timestamp']).strftime('%Y-%m-%d')


        # TODO: RESET ON FAILURE? HOW DO WE HANDLE FAILURES
        if df.loc[i, 'label'] == 1:  # Reset on failure
            df.loc[i, 'days_since_last_failure'] = 0
        elif df.loc[i, 'machine_id'] == df.loc[i-1, 'machine_id']:
            if current_date > prev_date:  # If the date has changed, increment the counter
                df.loc[i, 'days_since_last_failure'] = df.loc[i-1, 'days_since_last_failure'] + 1
            else:  # Same day, no increment
                df.loc[i, 'days_since_last_failure'] = df.loc[i-1, 'days_since_last_failure']

    return df

from datetime import datetime
def set_cumulative_month_failures(df):

    # Creating a function to compute cumulative_month_failures
    def calculate_cumulative_month_failures(row, df):
        # Get the timestamp of the current row
        current_timestamp = row['timestamp']

        # Filter the data for the same machine_id within the past month
        recent_failures = df[(df['label'] == 1)]

        # Convert timestamps to datetime objects and extract the YEAR-MONTH-DAY
        dates = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in recent_failures['timestamp']]

        # Get the unique days when failures occurred
        unique_failure_days = len(set(dates))
        #unique_failure_days = recent_failures['timestamp'].dt.date.nunique()

        return unique_failure_days

    # Apply the function to the DataFrame
    df['cumulative_month_failures'] = df.apply(calculate_cumulative_month_failures, axis=1, df=df)
    return df


def set_seasonality(df):

    df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour

    # Define seasonality based on shift hours
    #df['seasonality'] = df['hour'].apply(lambda x: 'day-shift' if 9 <= x < 18 else 'night-shift')
    df['seasonality'] = df['hour'].apply(lambda x: 0 if 9 <= x < 18 else 1)

    # Drop the temporary 'hour' column if not needed
    df.drop(columns=['hour'], inplace=True)
    return df

def set_changes(df):
    df['temperature_change'] = df.groupby('machine_id')['temperature'].diff().fillna(0)
    df['vibration_change'] = df.groupby('machine_id')['vibration'].diff().fillna(0)
    df['pressure_change'] = df.groupby('machine_id')['pressure'].diff().fillna(0)

    return df

def set_rolling_avg(df):
    def rolling_avg(df, time_window):
        rolling_avgs = []

        for index, row in df.iterrows():
            machine_id = row['machine_id']
            current_time = row['timestamp']
            start_time = find_timestamp_n_days(current_time, time_window)

            # Filter data for the same machine within the rolling window
            filtered_data = df[(df['machine_id'] == machine_id) &
                               (df['timestamp'] >= start_time) &
                               (df['timestamp'] <= current_time)]

            # Compute the rolling averages
            avg_temp = filtered_data['temperature'].mean() if not filtered_data.empty else 0
            avg_vib = filtered_data['vibration'].mean() if not filtered_data.empty else 0
            avg_press = filtered_data['pressure'].mean() if not filtered_data.empty else 0

            rolling_avgs.append((avg_temp, avg_vib, avg_press))

        return rolling_avgs

    # Compute rolling averages for 1-day, 2-day, and 3-day windows
    df[['rolling_avg_temp_1d', 'rolling_avg_vib_1d', 'rolling_avg_press_1d']] = rolling_avg(df, 1)
    df[['rolling_avg_temp_2d', 'rolling_avg_vib_2d', 'rolling_avg_press_2d']] = rolling_avg(df, 2)
    df[['rolling_avg_temp_3d', 'rolling_avg_vib_3d', 'rolling_avg_press_3d']] = rolling_avg(df, 3)

    return df

def enhance_data_for_model(sensor_data, old_documents):
    df = to_dataframe(sensor_data, old_documents)

    if 'label' not in df.columns:
        df['label'] = 0
    else:
        df['label'].fillna(0, inplace=True)
    df = set_days_since_last_failure(df)
    df = set_cumulative_month_failures(df)
    df = set_seasonality(df)
    df = set_changes(df)
    df = set_rolling_avg(df)

    return df

for message in consumer:
    sensor_data = message.value

    print(f"**Reading from kafka: {sensor_data}")
    old_documents = get_from_db(sensor_data['machine_id'],sensor_data['timestamp'])
    df = enhance_data_for_model(sensor_data, old_documents)
    df.drop(columns=["label"], inplace=True)
    latest_row = df.iloc[-1].to_dict()
    latest_row.pop('_id', None)
    print(f"**Storing to MongoDB: {latest_row}")
    collection.insert_one(latest_row)
    print(f"Saved: {latest_row}")
