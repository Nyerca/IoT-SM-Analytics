

days_since_last_failure,cumulative_month_failures,seasonality,temperature_change,vibration_change,pressure_change,rolling_avg_temp_1d,rolling_avg_vib_1d,rolling_avg_press_1d,rolling_avg_temp_2d,rolling_avg_vib_2d,rolling_avg_press_2d,rolling_avg_temp_3d,rolling_avg_vib_3d,rolling_avg_press_3d
121,0,0,0.3182066687482532,0.012639785841772166,0.7597253547223204,39.96780343859736,4.391276613685785,94.89963248613752,40.09371700957088,4.451652660084344,95.06724332116565,40.173078822469904,4.451031671746271,94.95563210146648

'days_since_last_failure' : to get that get the latest data: 
    if is lebel 1: then 0
    else
        if latest_data.timestamp (day) < today
            then days_since_last_failure = latest_data.days_since_last_failure + DELTA(latest_data.timestamp (day) - today)
        else
            then latest_data.days_since_last_failure

cumulative_month_failures get all data for given machine_id that has a timestamp >= today(-1 month)
    SUM failures

seasonality:
    no data required => get it from timestamp

temperature_change,vibration_change,pressure_change
    to get that get the latest data

rolling_avg_temp_1d,rolling_avg_vib_1d,rolling_avg_press_1d,rolling_avg_temp_2d,rolling_avg_vib_2d,rolling_avg_press_2d,rolling_avg_temp_3d,rolling_avg_vib_3d,rolling_avg_press_3d
    get data with timestamp >= (today - 3 days)



To fill all required informations I need to get all data for given machine_id that has a timestamp >= today(-1 month)




Pretrained model on:

day,temperature,label	,days_since_last_failure,cumulative_month_failures
1,34,0			,1,0
1,35,0			,1,0
2,33,0			,2,0
3,21,0			,3,0
4,40,1			,4,0
5,36,1			,5,0
6,39,1			,6,0
7,0,2	FAILURE (will be dropped we want just WARNING AREA (so lavel = 1))
8,33,0			,1,0
