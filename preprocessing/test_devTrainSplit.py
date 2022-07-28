"""Author: Rishav Sapahia"""

import pandas as pd
import os
import shutil
import time
two_cases = pd.read_csv('/home/ubuntu/QA_code/QA_application/Processed_Input_files/2_cases.csv').iloc[:, 1:]  # noqa

two_cases = two_cases.sample(frac=1, random_state=42)

sfact = int((0.05*len(two_cases))/2)
df_temp_1 = two_cases[two_cases['labels'] == 0][:sfact]
df_temp_2 = two_cases[two_cases['labels'] == 1][:sfact]
df_temp_3 = two_cases[two_cases['labels'] == 2][:sfact]
df_temp_4 = two_cases[two_cases['labels'] == 3][:sfact]

test_df = pd.concat([df_temp_1, df_temp_2, df_temp_3, df_temp_4])

test_y = test_df['labels']

df_train_val = two_cases[~two_cases['image'].isin(test_df['image'])]

source_files = '/home/ubuntu/EyePacs_Lenke_Dataset'
destination_folder = '/home/ubuntu/EyePacs_Lenke_Dataset_Division/Test'

image_names = list(test_df['image'])

for image in image_names:
    if(os.path.isfile(os.join(image, source_files))):
        shutil.move(os.join(source_files, image), destination_folder)

timestr = time.strftime("%Y%m%d-%H%M%S")
df_train_val.to_csv("/home/ubuntu/QA_code/QA_application/Train_Dev_Test_Split/2_cases_train_val_{0}.csv".format(timestr)) # noqa
test_df.to_csv("/home/ubuntu/QA_code/QA_application/Train_Dev_Test_Split/2_cases_test_{0}.csv".format(timestr))  # noqa
