# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 02:30:01 2018

Ultimate.Inc Data Science Challenge

logins.json analysis

@author: Jon
"""
# import datetime as dt
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

logins_filename = 'logins.json' 
ulti_filename   = 'ultimate_data_challenge.json'

#logins_df = pd.read_json(logins_filename) # works
logins_df = pd.read_json(logins_filename, convert_dates  = ['T'])

# ulti_data   = pd.read_json(ulti_filename) # Doesn't work

with open(ulti_filename) as f:
    ulti_data = json.load(f)

ulti_df = pd.DataFrame(ulti_data)


# Looking at the description of the logins df we know our first 15
# minute interval begins at 1970-01-01 20:00:00
# While our last 15 minute interval is at 1970-04-13 19:00:00

# We can use pd.date_range() to make our intervals

start_times = pd.date_range(start='1970-01-01 20:00:00', 
                          end = '1970-04-13 18:45:00', 
                          freq='15min')
# Make a list of interval end points
end_times = pd.date_range(start='1970-01-01 20:15:00', 
                          end = '1970-04-13 19:00:00', 
                          freq='15min')



# Now we want to make a list of dataframes

# logins_df_dict = {}
logins_df_list = []
interval_login_counts = []

earlyAM_login_list = []
morning_login_list = []
midday_login_list = []
afternoon_login_list = []
evening_login_list = []
night_login_list = []

for start, end in zip(start_times, end_times):
    # This line is for debugging
    print('Start Time:', start, ' End Time:', end)
    
    # Make the a new dataframe for each interval
    interval_df = logins_df[(logins_df['login_time'] > start) & 
                            (logins_df['login_time'] <= end)]
    interval_df.index = pd.RangeIndex(len(interval_df.index))
    
    # Get the count
    interval_count = len(interval_df)
    
    interval_login_counts.append(interval_count)
    
    # Assign it to the bigger list
    logins_df_list.append(interval_df)
    
    
    # Assign interval_df to time of day
    
    # Early AM: 00:00 through 04:59
    if (start.hour >= 0) & (start.hour < 5):
        earlyAM_login_list.append(interval_df)
    
    # Morning: 05:00 through 08:59
    elif (start.hour >= 5) & (start.hour < 9):
        morning_login_list.append(interval_df)
    
    # Midday: 09:00 through 11:59
    elif (start.hour >= 9) & (start.hour < 12):
        midday_login_list.append(interval_df)
    
    # Afternoon: 12:00 through 16:59
    elif (start.hour >= 12) & (start.hour < 17):
        afternoon_login_list.append(interval_df)
    
    # Evening: 17:00 through 19:59
    elif (start.hour >= 17) & (start.hour < 20):
        evening_login_list.append(interval_df)
    
    # Night: 20:00 through 23:59
    elif (start.hour >= 20) & (start.hour < 24):
        night_login_list.append(interval_df)
    
    
# Get a first idea
# plt.plot(interval_login_counts)
# plt.show()
# plt.clf()    
start_list = list(start_times)
t = np.arange(len(start_list))
interval_array = np.array(interval_login_counts)
# 103 days in the sample
daily_linspace = np.linspace(0, len(start_list), num = 103)

plt.figure(figsize=(12,6))
plt.bar(t, interval_array, color = 'blue')
plt.title('First Look: Ride Usage per 15 minutes')
plt.xticks(daily_linspace)
plt.xlabel('Time of day')
plt.ylabel('Rider count')
plt.ylim(0, 30)
plt.grid()
plt.show()
plt.clf()

print('\n\n\n')

# First day
day_one_array = interval_array[0:96]
day_t = np.arange(96)
day_linspace = np.linspace(0, 96, 25)

print('Day One Line Plot')
day_one_array = interval_array[0:96]
day_t = np.arange(96)
day_linspace = np.linspace(0, 96, 25)

plt.figure(figsize=(12,6))
plt.plot(day_t, day_one_array, color = 'orange')
plt.title('Day One: Ride Usage per 15 minutes')
plt.xticks(day_linspace)
plt.xlabel('nth 15 minute interval')
plt.ylabel('Rider count')
plt.ylim(0, 20)
plt.grid()
plt.show()
plt.clf()

print('Day One Bar Plot')


plt.figure(figsize=(12,6))
plt.bar(day_t, day_one_array, color = 'orange')
plt.title('Day One: Ride Usage per 15 minutes')
plt.xticks(day_linspace)
plt.xlabel('nth 15 minute interval')
plt.ylabel('Rider count')
plt.ylim(0, 20)
plt.grid()
plt.show()
plt.clf()

# First Week
print('\n\n\n')
print('Week One Line Plot')
week_one_array = interval_array[0:672]
week_t = np.arange(672)
week_linspace = np.linspace(0, 672, 8)

plt.figure(figsize=(12,6))
plt.plot(week_t, week_one_array, color = 'purple')
plt.title('Week One: Ride Usage per 15 minutes')
plt.xticks(week_linspace)
plt.xlabel('nth 15 minute interval (96 intervals in a day)')
plt.ylabel('Rider count')
plt.ylim(0, 35)
plt.grid()
plt.show()
plt.clf()


print('Week One Bar Plot')


plt.figure(figsize=(12,6))
plt.bar(week_t, week_one_array, color = 'purple')
plt.title('Week One: Ride Usage per 15 minutes')
plt.xticks(week_linspace)
plt.xlabel('nth 15 minute interval (96 intervals in a day)')
plt.ylabel('Rider count')
plt.ylim(0, 35)
plt.grid()
plt.show()
plt.clf()


# There are some clear patterns that may be part of a day night cycle
# Separating the times into times of day may elucidate things
# Concatenate the dataframes
earlyAM_login_df   = pd.concat(earlyAM_login_list)
morning_login_df   = pd.concat(morning_login_list)
midday_login_df    = pd.concat(midday_login_list)
afternoon_login_df = pd.concat(afternoon_login_list)
evening_login_df   = pd.concat(evening_login_list)
night_login_df     = pd.concat(night_login_list)


earlyAM_login_count   = len(earlyAM_login_df)
morning_login_count   = len(morning_login_df)
midday_login_count    = len(midday_login_df)
afternoon_login_count = len(afternoon_login_df)
evening_login_count   = len(evening_login_df)
night_login_count     = len(night_login_df)

time_titles = ['Early_AM', 'Morning', 'Midday', 'Afternoon', 'Evening', 'Night']
times_of_day_count_overall = [earlyAM_login_count,
                              morning_login_count,
                              midday_login_count,
                              afternoon_login_count,
                              evening_login_count,
                              night_login_count]

time_of_day_array = np.array(times_of_day_count_overall)
x = np.arange(6)

plt.figure(figsize=(8,4))
plt.bar(x, time_of_day_array, color = 'green')
plt.xticks(x,time_titles)
plt.title('Total ride usage by time of day')
plt.xlabel('Time of day')
plt.ylabel('Rider count')
plt.ylim(0, 30000)
plt.show()
plt.clf()

# Now we have a list of dataframes for later use
print('\n')
print('\n')
print('End of Logins data')
print('\n')
print('\n')

# In short, usage is lowest in the morning and rises through the day. 
# But users do not generally use the service in the evening rush hour
# But Nightlife usage is high. 
# This is probably a city where more people vacation and party than work.
# I.e This is possibly a tourist town about the nightlife.
# Shops often in the midday and early afternoon,
# And these shops stay open until the midnight hours. 


# Now we look at the ultimate data


# To Answer Part 2- Experiment and Metrics Design