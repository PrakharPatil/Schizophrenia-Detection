import os
import pandas as pd
import random
from glob import glob 
import csv

og_dir = os.getcwd()
os.chdir(rf'{og_dir}\Raw_Data')
Raw_data = os.getcwd()
csv_files = []
for item in os.listdir(Raw_data):
    if item.endswith('.csv'):
        csv_files.append(item)
# print(csv_files)
os.chdir(og_dir)
demographic = pd.read_csv("Data\demographic.csv")
Schizophrenic =[]
Healty =[]
Req_Cols =['subject', 'trial','condition','Fz','FCz','Cz','FC3','FC4','C3','C4','CP3','CP4',' time_ms']
for i, item in enumerate(list(demographic[" group"])):
    if item :
        Schizophrenic.append((demographic["subject"])[i])
    else:
        Healty.append((demographic["subject"])[i])
for i in range(len(csv_files)):
    df = pd.read_csv(f"Raw_Data\{csv_files[i]}")
    for x in range(0,5):
        rand = random.randrange(1,99)
        trial =(df['trial'] == rand)
        test = df['subject'][0]
        if df['subject'][0] in Schizophrenic:
            print(f'{test} : Schizophrenic')
            df[trial].to_csv(rf'Data\Schizophrenic\new_{csv_files[i]}',mode='a',index= False, columns=Req_Cols, header=None)
        elif df['subject'][0]in Healty:
            print(f'{test} : Healthy')
            df[trial].to_csv(rf'Data\Healthy\new_{csv_files[i]}',mode='a',index= False, columns=Req_Cols, header=None)

file = open(r'Data\columnLabels_filter.csv','r')
read = csv.reader(file)
data = list(read)
os.chdir(f'{os.getcwd()}\Data')
data_files = sorted(glob('*\*.csv'))
for i in range(len(data_files)):
    print(f'{data_files[i]}')
    for row in data:
        df = pd.read_csv(f'{data_files[i]}', header = None,low_memory=False)
        df.to_csv(f'{data_files[i]}', header= row,index=False )
file.close()

