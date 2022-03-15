# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import glob


# %%
import sys
#sys.path.insert(1, '/Users/swastik/ophthalmology/Project_Quality_Assurance/From_Bascom/Hold_out/') # noqa
sys.path.insert(1,'/Users/swastik/ophthalmology/Project_Quality_Assurance/Final_QA_FDA/QA_Application/training/Non_Distributed') # noqa
from import_packages.dataset_class_inference import Dataset
from import_packages.train_val_to_ids import inference_val_to_ids
from import_packages.checkpoint import load_checkpoint
import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet




# %%
path_dir = '/Users/swastik/ophthalmology/Project_Quality_Assurance/Final_QA_FDA/QA_application/training/Non_Distributed/Notebooks/Microsoft.csv' # noqa


# %%
source_dir = '/Volumes/My Book/Resized_Dataset' # noqa
#source_dir = '/Volumes/My Book/Small'


# %%
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


# %%

# %%
batch_size = 2
partition, labels = inference_val_to_ids(csv_file=path_dir, stratify_columns='labels') # noqa
test_set = Dataset(partition['test_set'],labels,root_dir=source_dir,test_transform=True) # noqa
test_loader = torch.utils.data.DataLoader(test_set,shuffle=True,pin_memory=True, num_workers =4, batch_size=batch_size)


# %%
model_transfer = EfficientNet.from_pretrained('efficientnet-b7')
n_inputs = model_transfer._fc.in_features
model_transfer._fc = nn.Linear(n_inputs, 2)


# %%
if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False


# %%
model = load_checkpoint(checkpoint_path='/Users/swastik/ophthalmology/Project_Quality_Assurance/Final_QA_FDA/Data_models/laced_thunder/binary_checkpoint_224.pt',model = model_transfer) # noqa


# %%
import csv 
def accuracy(model, loader,use_cuda=False):
    """Calculate Accuracy."""
    number = 0.0
    model.eval()
    for batch_idx, (data, target,id) in enumerate(loader):
        # print("data shape:", data.shape)
        # print("target shape:", target.shape)
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # update the average validation loss
        output = model(data)
        pred = torch.argmax(output, dim=1)
        number = pred.cpu().detach().tolist()
        # label_list.append(number)
        # name_list.append(list(id))
        with open('results.csv', 'a') as f: 
            write = csv.writer(f)
            write.writerow(list(id))
            write.writerow(number) 
            
    return None


# %%
accuracy(model, test_loader, use_cuda)



# %%



