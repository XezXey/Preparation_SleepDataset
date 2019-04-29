#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:39:50 2019

@author: puntawat
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import glob
import os
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns; sns.set(color_codes=True)
import errno

# Input subject folder
#subject_folder = sys.argv[1]
subject_folder = "Subject01"

#if len(sys.argv) == 1:
#    sys.exit("No subject input")

print("Data from : " + subject_folder)
path = './' + subject_folder + '*/All_Device_Preprocess/*.csv'

devices_filename = glob.glob(path)

# Removing raw filename and biosppy_preprocessed to ignore it from visualising
raw_filename = glob.glob('./' + subject_folder + '*/All_Device_Preprocess/*_raw.csv')
biosppy_filename = glob.glob('./' + subject_folder + '*/All_Device_Preprocess/*_biosppy.csv')
try:
    devices_filename.remove(raw_filename[0])
    devices_filename.remove(biosppy_filename[0])
    devices_filename.remove(biosppy_filename[1])
except IndexError or ValueError:
    print('--->Everything is fine. Nothing to be remove')


def find_filename(filename):
    # Function to find the device name in filename and use this as a key in dictionary of dataframe
    if 'applewatch' in filename:
        return 'applewatch'
    elif 'fitbit' in filename:
        return 'fitbit'
    elif 'emfitqs' in filename:
        return 'emfitqs'
    elif 'ticwatch' in filename:
        return 'ticwatch'
    elif 'polarh10' in filename:
        return 'polarh10'
    elif 'empatica' in filename:
        return 'empatica'
    elif 'biosignalsplux' in filename:
        return 'biosignalsplux'

devices_dict_df = {} # Storing the dataframe of each device
devices_list_df = [] # Storing the dataframe fo concaternate
for index, filename in enumerate(devices_filename):
    #print(index)
    #print(filename)
    devices_list_df.append(pd.read_csv(filename, index_col=0)) # Read and append dataframe in to list 
    devices_dict_df[find_filename(filename)] = pd.read_csv(filename, index_col=0) # Storing the dataframe into dictionary : {'device_name':dataframe}
# Concate the dataframe
devices_df = pd.concat(devices_list_df, ignore_index=True, sort=True) # sort = True : For retaining the current behavior and silence the warning, pass 'sort=True'.
list_hr_features = ['HR_applewatch', 'HR_polarh10', 'HR_empatica', 'HR_IBI_empatica', 'HR_fitbit', 'HR_emfitqs', 'HR_ticwatch', 'HR_biosignalsplux'] # List all heart rate features
# Filter the heart rate below 40 and over 200 out
print("--->Filter the outlier data...")
for each_hr_feature in list_hr_features:
    try :
        devices_df.loc[devices_df[each_hr_feature] > 200, each_hr_feature] = np.nan
        devices_df.loc[devices_df[each_hr_feature] < 40, each_hr_feature] = np.nan
    except KeyError:
        print('------>No ' + each_hr_feature + ' features found')

# Take millisecond part out and parse to datetime object(Downsampling for grouping the value
devices_df['Timestamp'] = devices_df['Timestamp'].apply(lambda each_time : dt.datetime.strptime(each_time.split('.')[0], '%Y-%m-%d %H:%M:%S').replace(microsecond=0))
devices_df = devices_df.sort_values(by=['Timestamp'], ascending=True).reset_index(drop = True) # Sort the value by Timestamp columns
cols = devices_df.columns.tolist() # Get the columns name
cols = cols[-1:] + cols[:-1] # Swap the name for rearange the columns 
devices_df = devices_df[cols] # Rearange the order of columns (Timestamp will be the first column)
interest_cols = ['HR_applewatch', 'HR_polarh10', 'HR_empatica', 'HR_IBI_empatica', 'HR_fitbit', 'HR_emfitqs', 'HR_ticwatch', 
                 'AX_empatica', 'AY_empatica', 'AZ_empatica', 'AX_empatica_abs', 'AY_empatica_abs', 'AZ_empatica_abs', 
                 'PA_lvl_AX_empatica', 'PA_lvl_AY_empatica', 'PA_lvl_AZ_empatica', 'PA_lvl_VectorA_empatica_encoded', 
                 'VectorA_empatica', 'HR_biosignalsplux']
devices_df = devices_df.loc[:, :].groupby(devices_df['Timestamp']).median() # Grouping the dataframe using timestamp(cut the ms part) and use mean to aggregate them
devices_df['Timestamp'] = devices_df.index.time


# Writing to csv file for only grouped
# # Also split into each activity
subject_folder = glob.glob(subject_folder + '*')[0]
#if subject_folder == []:
#    sys.exit("Cannot find that subject")

#subject_folder = 'Subject01_2019-1-16'

# Path for saving Grouped file
path_grouped = './' + subject_folder + '/All_Device_Grouped/'
# Trying to make directory if it's not exist
if not os.path.exists(os.path.dirname(path_grouped)):
    try:
        os.makedirs(os.path.dirname(path_grouped))
    except OSError as exc: #Guard against race condition
        if exc.errno != errno.EEXIST:
            raise
            
# Slicing into the resting and sleeping state
start_time_resting = devices_df['AX_empatica'].dropna().index[0] # Use AX_empatica to start the first resting(Once empatica start => the resting start)
end_time_resting = start_time_resting + dt.timedelta(minutes=30) # 30 minutes for resting
start_time_sleeping = end_time_resting + dt.timedelta(minutes=5) # 5 minutes gap for transition to sleeping
#end_time_sleeping = devices_df['HR_biosignalsplux'].dropna().index[-1]
end_time_sleeping = start_time_sleeping + dt.timedelta(minutes=90) # 90 minutes for sleeping
start_time_activity = devices_df['HR_polarh10'].dropna().index[0] # first record of polar is start point of intensity activity
end_time_activity = devices_df['HR_polarh10'].dropna().index[-1] # last record of polar is end point of intensity activity

# Splitting into each interval for saving seperately
devices_df_interval_resting = devices_df.loc[(devices_df['Timestamp'] > start_time_resting.time()) & (devices_df['Timestamp'] < end_time_resting.time())]
devices_df_interval_sleeping = devices_df.loc[(devices_df['Timestamp'] > start_time_sleeping.time()) & (devices_df['Timestamp'] < end_time_sleeping.time())]
devices_df_interval_activity = devices_df.loc[(devices_df['Timestamp'] > start_time_activity.time()) & (devices_df['Timestamp'] < end_time_activity.time())]

devices_df.to_csv(path_grouped + subject_folder + '_grouped_all_states.csv')
devices_df_interval_resting.to_csv(path_grouped + subject_folder + '_grouped_resting.csv')
devices_df_interval_sleeping.to_csv(path_grouped + subject_folder + '_grouped_sleeping.csv')
devices_df_interval_activity.to_csv(path_grouped + subject_folder + '_grouped_activity.csv')
