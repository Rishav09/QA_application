"""Author: Rishav Sapahia"""
import pandas as pd
import os
import shutil

path = '/home/ubuntu/QA_code/QA_application/Train_Dev_Test_Split/2_cases_train_val_20220727-181159.csv' # noqa

df_train_dev = pd.read_csv(path).iloc[:,1:]

image_names = list(df_train_dev['image'])

source_files = '/home/ubuntu/EyePacs_Lenke_Dataset'
destination_folder = '/home/ubuntu/EyePacs_Lenke_Dataset_Division/Train'

for image in image_names:
    shutil.move(os.path.join(source_files, image), destination_folder)
