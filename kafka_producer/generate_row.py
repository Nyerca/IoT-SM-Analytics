import numpy as np
from datetime import datetime, timedelta
import time

def initialize_machines(num_machines, num_days):
    machines = {}
    for machine_id in range(1, num_machines + 1):
        machines[machine_id] = {
            "temperature": np.random.uniform(40, 50),
            "vibration": np.random.uniform(4, 6),
            "pressure": np.random.uniform(90, 110),
            "failure_day": np.random.randint(30, num_days - 1)
        }
    return machines

def generate_row(machine_id, machines, day):
    previous_temp = machines[machine_id]["temperature"]
    previous_vibration = machines[machine_id]["vibration"]
    previous_pressure = machines[machine_id]["pressure"]

    print("PREV TEMP")
    print(previous_temp)

    failure_day = machines[machine_id]["failure_day"]

    temperature = previous_temp + np.random.uniform(-0.5, 0.5)
    vibration = previous_vibration + np.random.uniform(-0.2, 0.2)
    pressure = previous_pressure + np.random.uniform(-1, 1)

    # Apply warning signs 7 days before failure
    if failure_day - 7 <= day < failure_day:
        temperature += (day - (failure_day - 3)) * 1.5  # Gradual increase
        vibration += (day - (failure_day - 3)) * 0.7
        pressure += (day - (failure_day - 3)) * 7

    # Apply failure condition
    if day == failure_day:
        temperature += 5  # Sudden jump in temperature
        vibration += 2  # Sudden increase in vibration
        pressure += 15  # Sudden pressure spike

    machines[machine_id]["temperature"] = temperature
    machines[machine_id]["vibration"] = vibration
    machines[machine_id]["pressure"] = pressure

    #timestamp = datetime(2025, 1, 1) + timedelta(days=day, hours=np.random.randint(0, 24), minutes=np.random.randint(0, 60))
    timestamp = time.time()

    return {
        "machine_id": machine_id,
        "timestamp": timestamp,
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "failure_day": failure_day
    }

num_machines = 4
num_days = 200
machines = initialize_machines(num_machines, num_days)

print(machines)

generation_count = 0
def prepare_row(generation_count):
    machine_id = np.random.randint(1, num_machines + 1)
    print("Generating row for machine id: " + str(machine_id))
    day = 0  # Start at day 0

    new_row = generate_row(machine_id, machines, day)
    generation_count += 1
    if generation_count % np.random.choice([3, 4]) == 0:
        day += 1  # Increase the day every 3 or 4 generations
        generation_count = 0

    new_row.pop('failure_day', None)
    return new_row, generation_count

print(prepare_row(generation_count))
