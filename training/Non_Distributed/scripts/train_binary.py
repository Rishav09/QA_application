"""Author: Rishav Sapahia."""
import sys
sys.path.insert(1, '/home/ubuntu/QA_code/QA_application/training/Non_Distributed')
import torch
import torch.nn as nn
import timm
import wandb
import numpy as np
import os
import random
from import_packages.dataset_partition import split_equal_into_val, split_equal_into_test # noqa
from import_packages.dataset_class import Dataset
from import_packages.train_val_to_ids import train_val_to_ids
from import_packages.checkpoint import save_checkpoint

# sys.path.insert(1, '/Users/swastik/ophthalmology/Project_Quality_Assurance/Final_QA_FDA/Application/training/Non_Distributed') # noqa

 # noqa


# %%
# Deterministic Behavior
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

np.random.seed(seed)
random.seed(seed)
# CuDA Determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Wandb configuration
os.environ['WANDB_API_KEY'] = "344338e09b93dd41994593b9dd0fbcbe9407580c"
os.environ['WANDB_MODE'] = "online"
wandb.init(project="binary_qa")
config = wandb.config
# %%
config.batch_size = 1
temp_train,temp_valid= split_equal_into_val(csv_file='/home/ubuntu/QA_code/QA_application/Processed_Input_files/Split_folders/2_cases_train_val_20220727-181159.csv', stratify_colname='labels',no_of_classes=2) # noqa
temp_test = split_equal_into_test(csv_file='/home/ubuntu/QA_code/QA_application/Processed_Input_files/Split_folders/2_cases_test_20220727-181159.csv', stratify_colname='labels') # noqa
partition, labels=train_val_to_ids(temp_train, temp_valid, temp_test, stratify_columns='labels') # noqa
training_set = Dataset(partition['train_set'], labels, root_dir='/home/ubuntu/EyePacs_Lenke_Dataset_Division_64/train', train_transform=True) # noqa
validation_set = Dataset(partition['val_set'],labels,root_dir='/home/ubuntu/EyePacs_Lenke_Dataset_Division_64/train',valid_transform = True) # noqa
# test_set = Dataset(partition['test_set'],labels,root_dir='/home/ubuntu/EyePacs_Lenke_Dataset_Division_64/test',test_transform=True) # noqa
train_loader = torch.utils.data.DataLoader(training_set, shuffle=True, pin_memory=True, num_workers=32, batch_size=config.batch_size) # noqa
val_loader = torch.utils.data.DataLoader(validation_set,shuffle=True, pin_memory=True, num_workers=32, batch_size=config.batch_size) # noqa
# test_loader = torch.utils.data.DataLoader(test_set,shuffle=True,pin_memory=True, num_workers =32, batch_size=config.batch_size) # noqa

# %%
data_transfer = {'train': train_loader,
                 'valid': val_loader
                #  'test': test_loader
                 }

# model_transfer = EfficientNet.from_pretrained('efficientnet-b7')
model_transfer = timm.create_model('convnext_tiny_in22k', pretrained=True,num_classes=2) # noqa
# n_inputs = model_transfer._fc.in_features
# model_transfer._fc = nn.Linear(n_inputs, 2)


# for name, parameter in model_transfer.named_parameters():
#     parameter.requires_grad = False


# # %%
# update_params_name = ['_fc.weight', '_fc.bias', '_conv_head.weight']
# for name, parameter in model_transfer.named_parameters():
#     if name in update_params_name:
#         parameter.requires_grad = True


# %%
config.lr = 3e-4
weights = torch.tensor([1.5, 3.0])
optimizer = torch.optim.SGD(model_transfer.parameters(), lr=config.lr)
optimizer.zero_grad()
criterion_transfer = nn.BCEWithLogitsLoss()
#scheduler = ReduceLROnPlateau(optimizer,patience=4,factor=0.1,mode='min',verbose=True) # noqa
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=3e-8,max_lr=3e-2, step_size=8000,mode='triangular') # noqa


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_transfer.to(device)
criterion_transfer.to(device)
# %%
if torch.cuda.device_count() > 1:
    model_transfer = nn.DataParallel(model_transfer)
    criterion_transfer = criterion_transfer
model_transfer.to(device)



def train_model(model, loader, criterion, optimizer,  n_epochs, checkpoint_path): # noqa
    """Return trained model."""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        correct = 0.0
        total = 0.0
        accuracy = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loader['train']):
            # move to GPU
            if torch.cuda.is_available():
                data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking = True) # noqa
            optimizer.zero_grad()
            output = model(data)
            pred = torch.max(output, dim=1, keepdim=True)[0]
            target = torch.unsqueeze(target, dim=1)
            loss = criterion(pred.float(), target.float())
            loss.backward()
            optimizer.step()
            train_loss += ((1 / (batch_idx + 1)) * ((loss.data) - train_loss))
        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loader['valid']):
            # move to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            # update the average validation loss
            output = model(data)
            # Torch.max returns values and indices
            pred = torch.max(output, dim=1, keepdim=True)[0]
            preds = torch.max(output, dim=1, keepdim=True)[1]
            target = torch.unsqueeze(target, dim=1)
            loss = criterion(pred.float(), target.float())   # changes
            valid_loss += ((1 / (batch_idx + 1)) * ((loss.data) - valid_loss))
            correct += np.sum(np.squeeze(preds.eq(target.data.view_as(pred)).cpu().numpy())) # noqa
            total += data.size(0)
        accuracy = 100. * (correct/total)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \t Validation Accuracy: {:.6f} \t '.format( # noqa
            epoch,
            train_loss,
            valid_loss,
            accuracy,
            ))
        wandb.log({'Epoch': epoch, 'loss': train_loss,'valid_loss': valid_loss, 'Valid_Accuracy': accuracy}) # noqa
        # TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format( # noqa
                valid_loss_min, valid_loss))
            # Saving the  model
            state = {
                    'optimizer': optimizer.state_dict(),
                    'state_dict': model.state_dict(),
                    }
            save_checkpoint(state, is_best=True, checkpoint_path=None, best_model_path=checkpoint_path) # noqa
            valid_loss_min = valid_loss
    return model

train_model(model=model_transfer, loader=data_transfer, optimizer=optimizer, criterion=criterion_transfer,  n_epochs=50, checkpoint_path='/home/ubuntu/Saved_Models/binary_checkpoint_64.pt') # noqa
