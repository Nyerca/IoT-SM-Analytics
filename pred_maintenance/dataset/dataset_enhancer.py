import pandas as pd

# Load dataset
df = pd.read_csv("../dataset_mocked_sequential/predictive_maintenance_data.csv", parse_dates=["timestamp"])
df = df.sort_values(by=['machine_id', 'timestamp'])

# Sort values by machine_id and timestamp
pd.set_option('display.max_columns', None)
print(df)

# Compute "Time since last failure" (days since last label=1 per machine)
# Initialize the 'days_since_last_failure' column
df['days_since_last_failure'] = 1

# Iterate over the DataFrame to calculate the days since the last failure
for i in range(1, len(df)):
    # Check if the current row's machine_id is the same as the previous one and if the date has changed
    current_date = df.loc[i, 'timestamp'].date()
    prev_date = df.loc[i-1, 'timestamp'].date()

    if df.loc[i, 'label'] == 1:  # Reset on failure
        df.loc[i, 'days_since_last_failure'] = 0
    elif df.loc[i, 'machine_id'] == df.loc[i-1, 'machine_id']:
        if current_date > prev_date:  # If the date has changed, increment the counter
            df.loc[i, 'days_since_last_failure'] = df.loc[i-1, 'days_since_last_failure'] + 1
        else:  # Same day, no increment
            df.loc[i, 'days_since_last_failure'] = df.loc[i-1, 'days_since_last_failure']

print(df)

# Creating a function to compute cumulative_month_failures
def calculate_cumulative_month_failures(row, df):
    # Get the timestamp of the current row
    current_timestamp = row['timestamp']

    # Calculate the start of the past month
    start_of_month = current_timestamp - pd.DateOffset(months=1)

    # Filter the data for the same machine_id within the past month
    recent_failures = df[(df['machine_id'] == row['machine_id']) &
                         (df['timestamp'] >= start_of_month) &
                         (df['timestamp'] <= current_timestamp) &
                         (df['label'] == 1)]

    # Get the unique days when failures occurred
    unique_failure_days = recent_failures['timestamp'].dt.date.nunique()

    return unique_failure_days

# Apply the function to the DataFrame
df['cumulative_month_failures'] = df.apply(calculate_cumulative_month_failures, axis=1, df=df)




df['hour'] = df['timestamp'].dt.hour

# Define seasonality based on shift hours
#df['seasonality'] = df['hour'].apply(lambda x: 'day-shift' if 9 <= x < 18 else 'night-shift')
df['seasonality'] = df['hour'].apply(lambda x: 0 if 9 <= x < 18 else 1)

# Drop the temporary 'hour' column if not needed
df.drop(columns=['hour'], inplace=True)

# Display updated DataFrame
print(df)




df['temperature_change'] = df.groupby('machine_id')['temperature'].diff().fillna(0)
df['vibration_change'] = df.groupby('machine_id')['vibration'].diff().fillna(0)
df['pressure_change'] = df.groupby('machine_id')['pressure'].diff().fillna(0)











def rolling_avg(df, time_window):
    rolling_avgs = []

    for index, row in df.iterrows():
        machine_id = row['machine_id']
        current_time = row['timestamp']
        start_time = current_time - pd.Timedelta(days=time_window)

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

df.to_csv('sensor_data_parsed.csv', index=False)