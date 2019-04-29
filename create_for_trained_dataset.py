import sys
import glob
import os
import errno
from shutil import copy2

print("Creating dataset for training...", end='')
# Declare containing path
path_fortrainingmodel = './ForTrainingModel/'
path_resting = path_fortrainingmodel + 'Resting/'
path_sleeping = path_fortrainingmodel + 'Sleeping/'
path_intensity = path_fortrainingmodel + 'Intensity/'
path_allstates = path_fortrainingmodel + 'AllStates/'

# Put in in a list
paths = [path_fortrainingmodel, path_resting, path_sleeping, path_intensity, path_allstates]
grouped_path_states = ['All_Device_Grouped/Subject*_resting.csv', 'All_Device_Grouped/Subject*_sleeping.csv', 'All_Device_Grouped/Subject*_activity.csv', 
    'All_Device_Grouped/Subject*_all_states.csv']
copy_dst = [path_resting, path_sleeping, path_intensity, path_allstates]
# Creating the contained folder
for each_path in paths:
    if not os.path.exists(os.path.dirname(each_path)):
        try:
            os.makedirs(os.path.dirname(each_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


# Copy each grouped dataset in subject folders to ForTrainingModel/ folder
subject_folder = sorted(glob.glob('./Subject*/'))
#print(sorted(subject_folder))

for each_subject in subject_folder:
    for index, each_state in enumerate(grouped_path_states):
        grouped_file = glob.glob(each_subject + each_state)
        for each_file in grouped_file:
            copy2(each_file, copy_dst[index])
    
print('Done!') 
