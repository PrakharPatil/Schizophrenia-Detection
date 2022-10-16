import os
from glob import glob 
os.makedirs("Raw_Data", exist_ok='true')
os.makedirs("Data\Healthy", exist_ok='true')
os.makedirs("Data\Schizophrenic", exist_ok='true')
data_files_H = sorted(glob('Data\Healthy\*.csv'))
for f in data_files_H:
    os.remove(f)
data_files_S = sorted(glob('Data\Schizophrenic\*.csv'))
for f in data_files_S:
    os.remove(f)