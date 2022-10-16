import matplotlib.pyplot as plt
from glob import glob 
import pandas as pd
import os
og_dir=os.getcwd()
for foldername in ['Healthy','Schizophrenic']:
    os.chdir(fr'{og_dir}\Data\{foldername}')
    data_files = sorted(glob('*.csv'))
    for i in range(len(data_files)):
        print(f'{data_files[i]}')
        file = pd.read_csv(fr'{og_dir}\Data\{foldername}\{data_files[i]}')
        subj = file['subject'][0]
        for i in range(1,len(file['subject']),3072):
            temp = file.iloc[i:i+3071]
            trial = file['trial'][i]
            condition = file['condition'][i]
            subject = pd.to_numeric(temp['subject'],errors = "coerce").dropna()
            Fz = pd.to_numeric(temp['Fz'],errors = "coerce").dropna()
            FCz = pd.to_numeric(temp['FCz'],errors = "coerce").dropna()
            Cz = pd.to_numeric(temp['Cz'],errors = "coerce").dropna()
            FC3 = pd.to_numeric(temp['FC3'],errors = "coerce").dropna()
            FC4 = pd.to_numeric(temp['FC4'],errors = "coerce").dropna()
            C3 =  pd.to_numeric(temp['C3'],errors = "coerce").dropna()
            C4 =  pd.to_numeric(temp['C4'],errors = "coerce").dropna()
            CP3 = pd.to_numeric(temp['CP3'],errors = "coerce").dropna()
            CP4 = pd.to_numeric(temp['CP4'],errors = "coerce").dropna()
            time_ms = pd.to_numeric(temp[' time_ms'],errors = "coerce").dropna()
            os.makedirs(fr'{og_dir}\Project\{foldername}\{subj}\{trial}\{condition}',exist_ok='true')
            os.chdir(fr'{og_dir}\Project\{foldername}\{subj}\{trial}\{condition}')
            plt.specgram(Fz,1024)
            plt.savefig(f'Fz.png')
            plt.specgram(FC3,1024)
            plt.savefig(f'FC3.png')
            plt.specgram(FC4,1024)
            plt.savefig(f'FC4.png')
            plt.specgram(C3,1024)
            plt.savefig(f'C3.png')
            plt.specgram(FCz,1024)
            plt.savefig(f'FCz.png')
            plt.specgram(Cz,1024)
            plt.savefig(f'Cz.png')
            plt.specgram(C4,1024)
            plt.savefig(f'C4.png')
            plt.specgram(CP3,1024)
            plt.savefig(f'CP3.png')
            plt.specgram(CP4,1024)
            plt.savefig(f'CP4.png')