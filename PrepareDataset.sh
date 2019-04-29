#!/bin/bash
# Preparing the dataset including
# 1. PreprocessData.py : For preprocess a data
# 2. Grouping_all_devices.py : For Grouping data and features into each activities
# 3. create_for_trained_dataset : For Moving the dataset seperate by activities into folder for easily import by training process


N_SUBJECT=21 # Define number of subject

# Iterate over each subject to PreprocessData
for ((subject=1; subject<=N_SUBJECT; subject++)); do
    SUBJECT_PTR=$(printf "Subject%02d" $subject)
    python ./PreprocessData.py $SUBJECT_PTR
done

# Iterate over each subject to Grouping features and data into each activites
for ((subject=1; subject<=N_SUBJECT; subject++)); do
    SUBJECT_PTR=$(printf "Subject%02d" $subject)
    python ./Grouping_all_devices.py $SUBJECT_PTR
done

# Create a fro_trained_dataset folder 
python ./create_for_trained_dataset.py
