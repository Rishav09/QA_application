"""Author: Rishav Sapahia."""
import sys
#sys.path.insert(1, '/Users/swastik/ophthalmology/Project_Quality_Assurance/From_Bascom/Hold_out/') # noqa
sys.path.insert(1,'/Users/swastik/ophthalmology/Project_Quality_Assurance/Final_QA_FDA/Application/training/Non_Distributed') # noqa
from import_packages.dataset_class import Dataset
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
path_dir = '/Users/swastik/ophthalmology/Project_Quality_Assurance/Final_QA_FDA/data_models/chocolate_feather/df_test.csv' # noqa


# %%
df = pd.read_csv(path_dir).iloc[:, 1:]

# %%
source_dir = '/Users/swastik/ophthalmology/Project_Quality_Assurance/Final_QA_FDA/Data_Staging/Combined_Dataset_224' # noqa


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
batch_size = 2
partition, labels = inference_val_to_ids(csv_file=path_dir, stratify_columns='labels') # noqa
test_set = Dataset(partition['test_set'],labels,root_dir=source_dir,test_transform=True) # noqa
test_loader = torch.utils.data.DataLoader(test_set,shuffle=True,pin_memory=True, num_workers =4, batch_size=batch_size) # noqa


# %%
model_transfer = EfficientNet.from_pretrained('efficientnet-b7')
n_inputs = model_transfer._fc.in_features
model_transfer._fc = nn.Linear(n_inputs, 4)


# %%
weights = torch.tensor([0.002, 0.0002, 0.0002, 0.0002])
optimizer = torch.optim.SGD(model_transfer.parameters(), lr=3e-4)
criterion_transfer = nn.CrossEntropyLoss(weight=weights, reduction='mean')
scheduler = ReduceLROnPlateau(
            optimizer,
            patience=4,
            factor=0.1,
            mode='min',
            verbose=True
        )


# %%
if torch.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False


# %%
model = load_checkpoint(checkpoint_path='/Users/swastik/ophthalmology/Project_Quality_Assurance/From_Bascom/models/checkpoint2_600.pt',model = model_transfer) # noqa


# %%
def accuracy(model, loader, criterion, use_cuda=False):
    """Calculate Accuracy."""
    test_accuracy = 0.0
    test_correct = 0.0
    test_total = 0.0
    model.eval()
    for batch_idx, (data, target) in enumerate(loader):
        # print(batch_idx)
        # print("data shape:", data.shape)
        # print("target shape:", target.shape)
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # update the average validation loss
        output = model(data).squeeze()
        pred = torch.argmax(output, dim=1)
        test_correct += np.sum(np.squeeze(pred.eq(target.view_as(pred)).cpu().numpy())) # noqa
        test_total += data.size(0)
    test_accuracy = 100. * (test_correct/test_total)
    print(test_accuracy)
    return None


accuracy(model, test_loader, criterion_transfer, use_cuda)
