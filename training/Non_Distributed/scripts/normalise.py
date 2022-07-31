"""To reduce the dataset to mean 0 and standard deviation 1."""
import sys
sys.path.insert(1, '/home/ubuntu/QA_code/QA_application/training/Non_Distributed') # noqa
from import_packages.dataset_partition import  split_equal_into_val, split_equal_into_test # noqa
from import_packages.dataset_class import Dataset
from import_packages.train_val_to_ids import train_val_to_ids
import os
import numpy as np
import random
import torch
from tqdm import tqdm
sys.path.insert(1, '/home/ubuntu/QA_code/QA_application/training/Non_Distributed') # noqa


# sys.path.insert(1, '/Users/swastik/ophthalmology/Project_Quality_Assurance/Final_QA_FDA/Application/training/Non_Distributed') # noqa

#sys.path.insert(1, '/home/ubuntu/QA_code/Final_QA_FDA/QA_application/training/Non_Distributed') # noqa

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)
# CuDA Determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = 8
temp_train,temp_valid= split_equal_into_val(csv_file='/home/ubuntu/QA_code/QA_application/Processed_Input_files/Split_folders/2_cases_train_val_20220727-181159.csv', stratify_colname='labels',no_of_classes=2) # noqa
temp_test = split_equal_into_test(csv_file='/home/ubuntu/QA_code/QA_application/Processed_Input_files/Split_folders/2_cases_test_20220727-181159.csv', stratify_colname='labels') # noqa
partition, labels=train_val_to_ids(temp_train, temp_test, temp_valid, stratify_columns='labels') # noqa
training_set = Dataset(partition['train_set'], labels, root_dir='/home/ubuntu/EyePacs_Lenke_Dataset_Division_448/train', train_transform=True) # noqa
validation_set = Dataset(partition['val_set'],labels,root_dir='/home/ubuntu/EyePacs_Lenke_Dataset_Division_448/train',valid_transform = False) # noqa
test_set = Dataset(partition['test_set'],labels,root_dir='/home/ubuntu/EyePacs_Lenke_Dataset_Division_448/test',test_transform=False) # noqa
train_loader = torch.utils.data.DataLoader(training_set, shuffle=True, pin_memory=True, num_workers=1, batch_size=batch_size) # noqa
val_loader = torch.utils.data.DataLoader(validation_set,shuffle=True, pin_memory=True, num_workers=1, batch_size=batch_size) # noqa


# %%
data_transfer = {'train': train_loader,
                 'valid': val_loader
                 }
train_mean = []
train_std = []

for i,image in enumerate(tqdm(train_loader, 0)):
    numpy_image = image[0].numpy()
    batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
    batch_std = np.std(numpy_image, axis=(0, 2, 3))
    train_mean.append(batch_mean)
    train_std.append(batch_std)  
train_mean = torch.tensor(np.mean(train_mean, axis=0))
train_std = torch.tensor(np.mean(train_std, axis=0))

print('Mean:', train_mean,file=open('/home/ubuntu/QA_code/QA_application/preprocessing/Mean_std3.txt', 'a')) # noqa
print('Std Dev:', train_std,file=open('/home/ubuntu/QA_code/QA_application/preprocessing/Mean_std3.txt', 'a')) # noqa


# val_mean = []
# val_std = []

# for i,image in enumerate(val_loader, 0):
#     numpy_image = image[0].numpy()
#     batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
#     batch_std = np.std(numpy_image, axis=(0, 2, 3))

#     val_mean.append(batch_mean)
#     val_std.append(batch_std)

# val_mean = torch.tensor(np.mean(val_mean, axis=0))
# val_std = torch.tensor(np.mean(val_std, axis=0))
# print('Val_Mean:', val_mean, file=open('./Mean_std.txt','a'))
# print('Val_Std Dev:', val_std, file=open('./Mean_std.txt','a'))




