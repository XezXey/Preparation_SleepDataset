#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:23:21 2019
Preprocess the device data file
@author: puntawat
"""
# Import libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import timeit
import errno
from freedson_adult_1998 import generate_60s_epoch, freedson_adult_1998

runtime_start = timeit.default_timer()

#if len(sys.argv) == 1:
#    sys.exit("No subject input")
subject_folder = sys.argv[1]
#subject_folder = "Subject01"
subject_name = subject_folder
print("Start...Preprocessing all devices file : {0}".format(subject_name))
subject_folder = glob.glob(subject_folder + '*')[0]
#if subject_folder == []:
#    sys.exit("Cannot find that subject")

# Create path to save preprocess file
path = './' + subject_folder + '/' + 'All_Device_Preprocess/'
# Trying to make directory if it's not exist
if not os.path.exists(os.path.dirname(path)):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as exc: #Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

############
# Empatica #
############
def preprocess_acc(each_acc_df):
    # Convert AX/AY/AZ to an Vector of accelerometer
    each_acc_df['ax'] = each_acc_df['ax'] * 1/128
    each_acc_df['ay'] = each_acc_df['ay'] * 1/128
    each_acc_df['az'] = each_acc_df['az'] * 1/128
    each_acc_df['AX_empatica_abs'] = np.abs(each_acc_df['ax'])
    each_acc_df['AY_empatica_abs'] = np.abs(each_acc_df['ay'])
    each_acc_df['AZ_empatica_abs'] = np.abs(each_acc_df['az'])

    each_acc_df['VectorA_empatica'] = np.sqrt(np.square(each_acc_df['ax'].astype(np.float64)) + np.square(each_acc_df['ay'].astype(np.float64)) + np.square(each_acc_df['az'].astype(np.float64)))
    # Calculate PAL and merge back to original acc dataframe
    ax_pa_lvl = pd.DataFrame(freedson_adult_1998({'time' : each_acc_df.TS_Machine.values, 'values':each_acc_df['ax'].values}, sampling_rate))
    ax_pa_lvl.rename(columns={'time':'TS_Machine', 'PA_Level':'PA_lvl_AX_empatica'}, inplace=True)
    each_acc_df = each_acc_df.merge(ax_pa_lvl[['TS_Machine', 'PA_lvl_AX_empatica']], on='TS_Machine', how='left').sort_values(by=['TS_Machine'], ascending=True).reset_index(drop = True)

    ay_pa_lvl = pd.DataFrame(freedson_adult_1998({'time' : each_acc_df.TS_Machine.values, 'values':each_acc_df['ay'].values}, sampling_rate))
    ay_pa_lvl.rename(columns={'time':'TS_Machine', 'PA_Level':'PA_lvl_AY_empatica'}, inplace=True)
    each_acc_df = each_acc_df.merge(ay_pa_lvl[['TS_Machine', 'PA_lvl_AY_empatica']], on='TS_Machine', how='left').sort_values(by=['TS_Machine'], ascending=True).reset_index(drop = True)

    az_pa_lvl = pd.DataFrame(freedson_adult_1998({'time' : each_acc_df.TS_Machine.values, 'values':each_acc_df['az'].values}, sampling_rate))
    az_pa_lvl.rename(columns={'time':'TS_Machine', 'PA_Level':'PA_lvl_AZ_empatica'}, inplace=True)
    each_acc_df = each_acc_df.merge(az_pa_lvl[['TS_Machine', 'PA_lvl_AZ_empatica']], on='TS_Machine', how='left').sort_values(by=['TS_Machine'], ascending=True).reset_index(drop = True)

    vectora_pa_lvl = pd.DataFrame(freedson_adult_1998({'time' : each_acc_df.TS_Machine.values, 'values':each_acc_df['VectorA_empatica'].values}, sampling_rate))
    vectora_pa_lvl.rename(columns={'time':'TS_Machine', 'PA_Level':'PA_lvl_VectorA_empatica'}, inplace=True)
    each_acc_df = each_acc_df.merge(vectora_pa_lvl[['TS_Machine', 'PA_lvl_VectorA_empatica']], on='TS_Machine', how='left').sort_values(by=['TS_Machine'], ascending=True).reset_index(drop = True)
    
    # Encoding the PA Level into numerica values
    each_acc_df['PA_lvl_AX_empatica_encoded'] = each_acc_df['PA_lvl_AX_empatica'].map({'Sedentary':1, 'Light':2, 'Moderate':3, 'Vigorous':4, 'Very Vigorous':5})
    each_acc_df['PA_lvl_AY_empatica_encoded'] = each_acc_df['PA_lvl_AY_empatica'].map({'Sedentary':1, 'Light':2, 'Moderate':3, 'Vigorous':4, 'Very Vigorous':5})
    each_acc_df['PA_lvl_AZ_empatica_encoded'] = each_acc_df['PA_lvl_AZ_empatica'].map({'Sedentary':1, 'Light':2, 'Moderate':3, 'Vigorous':4, 'Very Vigorous':5})
    each_acc_df['PA_lvl_VectorA_empatica_encoded'] = each_acc_df['PA_lvl_VectorA_empatica'].map({'Sedentary':1, 'Light':2, 'Moderate':3, 'Vigorous':4, 'Very Vigorous':5})

    return each_acc_df
            
sampling_rate = 32 # PAL sampling rate
try:
    empatica_list_df = [] # Store all empatica dataframe
    empatica_list_filename = [] # Store all empatica filename
    empatica_dict_df = {} # Create dict of dataframe : {'filename' : dataframe}
    empatica_filename = ['Acc', 'Batt', 'Bvp', 'Gsr', 'Hr', 'Ibi', 'Tag', 'Tmp']
    for fn in empatica_filename: 
        # Iterate over features data
        # Listing and add all filename into list for merge and concat
        # Matching the word in filename, list and add it into list    
        empatica_list_filename = glob.glob('./' + subject_folder + '*/Empatica/subject*' + fn + '*')

        # Iterate in filename for each feature
        for each_fn in range(len(empatica_list_filename)):
            # Read all file and append into dataframe
            empatica_list_df.append(pd.read_csv(empatica_list_filename[each_fn]))
        
        if fn == 'Acc': # Preprocess Acc and calculate PAL
            for index_df, each_acc_df in enumerate(empatica_list_df):
                empatica_list_df[index_df] = preprocess_acc(each_acc_df)
                
        # Concaternating
        if len(empatica_list_df) == 1:
            # Have only 1 file so assign it normally
            empatica_dict_df[fn] = empatica_list_df[0]
        else:
            # More than 1 file ===> using concat
            empatica_dict_df[fn] = pd.concat(empatica_list_df, ignore_index = True)
            
        # Sorting the dataframe using Timestamp columns
        empatica_dict_df[fn] = empatica_dict_df[fn].sort_values(by=['TS_Machine'], ascending=True).reset_index(drop = True)
        # Delete the ['TS_Empt'] out for preventing duplicate Timestamp
        empatica_dict_df[fn].pop('TS_Empt')
        # Free the list to get next features
        empatica_list_df = []
        
    # Merge empatica df with BVP_empatica for using the highest sampling rate    
    empatica_merged_df = empatica_dict_df['Bvp'] # Initial merged dataframe to 'Bvp' feature
    empatica_merge_column = empatica_filename.remove('Bvp') # Remove this column name out to prevent the duplicate columns
    for feature in empatica_filename:
        # Iterate over feature and use it as key to get the value in empatica_dict_df and merge all df together and sort by Timestamp
        empatica_merged_df = empatica_merged_df.merge(empatica_dict_df[feature], on='TS_Machine', how='outer').sort_values(by=['TS_Machine'], ascending=True).reset_index(drop = True)

    # Parse Unix Timestamp to datetime object
    empatica_merged_df = empatica_merged_df[pd.notnull(empatica_merged_df['TS_Machine'])]
    empatica_merged_df['TS_Machine'] = empatica_merged_df['TS_Machine'].apply(lambda each_time : dt.datetime.fromtimestamp(each_time))
    # Rename the columnns
    empatica_merged_df.rename(columns={'TS_Machine':'Timestamp', 'bvp':'BVP_empatica', 
                                       'ax':'AX_empatica', 'ay':'AY_empatica', 'az':'AZ_empatica', 
                                       'ibi':'IBI_empatica', 'gsr':'GSR_empatica', 'hr':'HR_empatica', 'tag':'TAG_empatica', 
                                       'tmp':'TEMP_empatica', 'batt':'BATT_empatica'}, inplace=True)

    #Calculate Heart rate from IBI
    empatica_merged_df['HR_IBI_empatica'] = 60/empatica_merged_df['IBI_empatica']

    # Saving the empatica preprocessed file
    empatica_merged_df.to_csv(path + subject_folder + '_empatica.csv')

except ValueError :
    print("--->There's no empatica record files")
except Exception as exception:
    print(str(exception))

###########
# EmfitQS #
###########
emfitqs_list_df = []
# Listing the file in EmfitQS folders
emfitqs_list_filename = glob.glob('./' + subject_folder + '*/EmfitQS/subject*')

# Read and append the dataframe in to list and concaternate them
for fn in emfitqs_list_filename:
    emfitqs_list_df.append(pd.read_csv(fn))
if len(emfitqs_list_df) > 1:
    emfitqs_concat = pd.concat(emfitqs_list_df,ignore_index=True)
else:
    emfitqs_concat = emfitqs_list_df[0]

# Rename the columns
emfitqs_concat.columns = list(map(lambda each_col : each_col + '_emfitqs', emfitqs_concat.columns))
emfitqs_concat = emfitqs_concat.rename(columns={'timestamp_from_machine_emfitqs':'Timestamp', })
# Sorting the value by timestamp and parse to datetime object
emfitqs_concat = emfitqs_concat.sort_values(by=('Timestamp'), ascending=True).reset_index(drop=True)
emfitqs_concat['Timestamp'] = emfitqs_concat['Timestamp'].apply(lambda each_time : dt.datetime.fromtimestamp(each_time))
emfitqs_concat.to_csv(path + subject_folder + '_emfitqs.csv')


############
# Ticwatch #
############
ticwatch_list_df = [] # for storing the dataframe obj
# list the filename from ticwatch folder
ticwatch_list_filename = glob.glob('./' + subject_folder + '*/Ticwatch/subject*.csv')
# Read and append the dataframe to list for concaternate
if len(ticwatch_list_filename) != 0:
    for fn in ticwatch_list_filename:
        ticwatch_list_df.append(pd.read_csv(fn))
        
    if len(ticwatch_list_df) > 1:
        ticwatch_concat = pd.concat(ticwatch_list_df, ignore_index = True)
    else:
        ticwatch_concat = ticwatch_list_df[0]

    ticwatch_concat.pop('end') # Delete ['end'] columns
    # Sort and rename the columns
    ticwatch_concat = ticwatch_concat.sort_values(by=['start'])
    ticwatch_concat.rename(columns={'start':'Timestamp', 'value':'HR_ticwatch'}, inplace=True)
    ticwatch_concat['Timestamp'] = ticwatch_concat['Timestamp']/1000
    ticwatch_concat['Timestamp'] = ticwatch_concat['Timestamp'].apply(lambda each_time : dt.datetime.fromtimestamp(each_time))
    ticwatch_concat.to_csv(path + subject_folder + '_ticwatch.csv')

############
# PolarH10 #
############
temp_time = [] # Polar have a start time and sampling rates So need to create the timestamp by start_time + sampling_rate
polarh10_list_df = [] # Store dataframes
polarh10_list_filename = glob.glob('./'  + subject_folder + '*/PolarH10/subject*.csv') # Listing the filename to append
for fn in polarh10_list_filename:
    #Processing Timestamp
    polarh10_description_df = pd.read_csv(fn, nrows=1)
    date = polarh10_description_df['Date'][0]
    start_time = polarh10_description_df['Start time'][0]
    polarh10_data_df = pd.read_csv(fn, skiprows=2)
    start_time_obj = dt.datetime.strptime(date + '_' + start_time, '%d-%m-%Y_%H:%M:%S')
    polarh10_data_df = polarh10_data_df.loc[:, ['Time', 'HR (bpm)']]
    each_time_obj = start_time_obj
    for i in range(0, len(polarh10_data_df)):
        each_time_obj = each_time_obj + dt.timedelta(seconds=1) # Adding 1s for 1hz sampling rate from polarh10
        temp_time.append(each_time_obj)
    # Add the timestamp columns into dataframe
    polarh10_data_df['Timestamp'] = temp_time
    # Free the list for next file
    temp_time = []
    # Append the dataframe to list
    polarh10_list_df.append(polarh10_data_df)

# Concaternate the dataframe in list
if len(polarh10_data_df) > 1 :
    polarh10_concat = pd.concat(polarh10_list_df, ignore_index=True)
else:
    polarh10_concat = polarh10_list_df[0]
polarh10_concat.pop('Time') # Pop ['Time'] columns 
polarh10_concat = polarh10_concat.loc[:, ['Timestamp', 'HR (bpm)']]
polarh10_concat.rename(columns={'HR (bpm)':'HR_polarh10'}, inplace=True) # Rename the timestamp columns
polarh10_concat = polarh10_concat.sort_values(by=('Timestamp'), ascending=True).reset_index(drop=True)
polarh10_concat.to_csv(path + subject_folder + '_polarh10.csv')

##########
# Fitbit #
##########
# List the fitbit files
fitbit_list_filename = glob.glob('./'  + subject_folder + '*/Fitbit/subject*.csv')
fitbit_list_df = [] # Storing a dataframe 
for fn in fitbit_list_filename:
    fitbit_df = pd.read_csv(fn, index_col=0)
    fitbit_df['Timestamp'] = fitbit_df['Timestamp'].apply(lambda each_timestamp : dt.datetime.strptime(each_timestamp, '%Y-%m-%d_%H:%M:%S'))
    # Decoded the onehot-encoder like data format
    fitbit_df['PA_lvl_fitbit'] = fitbit_df[['Sedentary_fitbit', 'LightlyActive_fitbit', 'FairlyActive_fitbit', 'VeryActive_fitbit']].idxmax(1)
    # Encoded to 1 column of PA_lvl_fitbit_encoded
    fitbit_df['PA_lvl_fitbit_encoded'] = fitbit_df['PA_lvl_fitbit'].map({'Sedentary_fitbit':1, 'LightlyActive_fitbit':2, 
             'FairlyActive_fitbit':3, 'VeryActive_fitbit':4})
    fitbit_list_df.append(fitbit_df)
fitbit_df = pd.concat(fitbit_list_df)
fitbit_df.to_csv(path + subject_folder + '_fitbit.csv')
    
###############
# AppleWatch4 #
###############
applewatch_df = pd.read_csv(glob.glob('./' + subject_folder + '*/Apple*/subject*.csv')[0]) # List the filename
applewatch_df.rename(columns={'time': 'Timestamp', 'hr':'HR_applewatch'}, inplace=True) # Rename the columns name
applewatch_df['Timestamp'] = applewatch_df['date'] + '_' + applewatch_df['Timestamp']
applewatch_df.pop('date') # Delete date columns
applewatch_df.pop('timezone') # Delete timezone columns
applewatch_df['Timestamp'] = applewatch_df['Timestamp'].apply(lambda each_timestamp : dt.datetime.strptime(each_timestamp, '%Y-%m-%d_%H:%M:%S')) # Cast timestamp to datetime object
applewatch_df.to_csv(path + subject_folder + '_applewatch.csv') # Write preprocessed data to csv file

runtime_stop = timeit.default_timer()
print("Finishing...Preprocessing all devices file : " + subject_name + ' (Runtime : ' + str(runtime_stop - runtime_start) + ' s)') 
