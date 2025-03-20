import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample dataset
np.random.seed(42)
num_machines = 12  # Number of machines
num_days = 200  # Number of days of data per machine
samples_per_day = np.random.randint(3, 6, size=num_days)  # Random 3 to 5 samples per day

data = []
columns = ["machine_id", "timestamp", "temperature", "vibration", "pressure", "label"]

total_rows = 0

for machine in range(1, num_machines + 1):
    failure_day = np.random.randint(30, num_days - 1)  # Random failure day for each machine
    base_temp = np.random.uniform(40, 50)
    base_vibration = np.random.uniform(4, 6)
    base_pressure = np.random.uniform(90, 110)

    for day in range(num_days):
        if np.random.rand() < 0.1:  # 10% chance of missing data for the day
            continue

        timestamp_base = datetime(2025, 1, 1) + timedelta(days=day)
        previous_temp = base_temp
        previous_vibration = base_vibration
        previous_pressure = base_pressure

        for _ in range(samples_per_day[day]):
            timestamp = timestamp_base + timedelta(hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60))

            # Simulate workload variation (increase or decrease in values)
            load_factor = np.sin(day / 10)  # Alternating load conditions
            temperature = previous_temp + np.random.uniform(-0.5, 0.5)  # Small variations between samples
            vibration = previous_vibration + np.random.uniform(-0.2, 0.2)
            pressure = previous_pressure + np.random.uniform(-1, 1)
            label = 0  # Default normal operation

            previous_temp = temperature
            previous_vibration = vibration
            previous_pressure = pressure

            # Apply warning signs 3 days before failure
            if failure_day - 7 <= day < failure_day:
                temperature += (day - (failure_day - 3)) * 1.5  # Gradual increase
                vibration += (day - (failure_day - 3)) * 0.7
                pressure += (day - (failure_day - 3)) * 7
                label = 1

            # Apply failure condition
            if day == failure_day:
                temperature += 5  # Sudden jump in temperature
                vibration += 2  # Sudden increase in vibration
                pressure += 15  # Sudden pressure spike
                label = 2

            data.append([machine, timestamp, temperature, vibration, pressure, label])
            total_rows += 1

            if total_rows >= 10000:
                break

        if total_rows >= 10000:
            break

    if total_rows >= 10000:
        break

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("predictive_maintenance_data.csv", index=False)

print(df.head(20))  # Show first 20 rows