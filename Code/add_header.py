import pandas as pd
from glob import glob 
import csv 
import os

file = open('Data\columnLabels.csv','r')
read = csv.reader(file)
data = list(read)
os.chdir(f'{os.getcwd()}\Raw_Data')
data_files = sorted(glob('*.csv'))
for i in range(len(data_files)):
    print(f'{data_files[i]}')
    for row in data:
        df = pd.read_csv(f'{data_files[i]}', header = None,low_memory=False)
        df.to_csv(f'{data_files[i]}', header= row,index=False )
file.close()

