import pandas as pd
from glob import glob 
import os

time = pd.read_csv(r"Data\time.csv")
os.chdir(f'{os.getcwd()}\Raw_Data')
data_files = sorted(glob('*.csv'))
for i in range(len(data_files)):
    print(f'{data_files[i]}')
    df = pd.read_csv(f'{data_files[i]}', low_memory=False)
    output = pd.merge(df, time, 
                        on='sample', 
                        how='left',validate='many_to_one')
    output.to_csv(f'{data_files[i]}',index=False )
print('Hello')