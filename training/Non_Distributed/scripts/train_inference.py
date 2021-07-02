""" Author: Rishav Sapahia """
# %%
import numpy as np
import os
import random

# %%
from import_packages.dataset_partition import split_equal_into_val_test
from import_packages.dataset_class import Dataset
from import_packages.train_val_to_ids import train_val_to_ids
import torch
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

# %%
# Deterministic Behavior
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
batch_size = 8
temp_train,temp_valid, temp_test = split_equal_into_val_test(csv_file='/opt/dkube/newinput/Final_Input_With_normalized_labels.csv', stratify_colname='labels') # noqa
partition, labels=train_val_to_ids(temp_train, temp_test, temp_valid, stratify_columns='labels') # noqa
training_set = Dataset(partition['train_set'], labels, root_dir='/opt/dkube/input/Mod_AEON_data', train_transform=False) # noqa
validation_set = Dataset(partition['val_set'],labels,root_dir='/opt/dkube/input/Mod_AEON_data',valid_transform = False) # noqa
test_set = Dataset(partition['test_set'],labels,root_dir='/opt/dkube/input/Mod_AEON_data',test_transform=True) # noqa
train_loader = torch.utils.data.DataLoader(training_set, shuffle=True, pin_memory=True, num_workers=0, batch_size=batch_size) # noqa
val_loader = torch.utils.data.DataLoader(validation_set,shuffle=True, pin_memory=True, num_workers=0, batch_size=batch_size) # noqa
test_loader = torch.utils.data.DataLoader(test_set,shuffle=True,pin_memory=True, num_workers =0, batch_size=batch_size) # noqa

# %%
data_transfer = {'train': train_loader,
                 'valid': val_loader,
                 'test': test_loader
                 }

model_transfer = EfficientNet.from_pretrained('efficientnet-b7')
n_inputs = model_transfer._fc.in_features
model_transfer._fc = nn.Linear(n_inputs, 4)


# %%
# %%
for name, parameter in model_transfer.named_parameters():
    parameter.requires_grad = False


# %%
update_params_name = ['_fc.weight', '_fc.bias', '_conv_head.weight']
for name, parameter in model_transfer.named_parameters():
    if name in update_params_name:
        parameter.requires_grad = True


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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_transfer.to(device)

# %%
if torch.cuda.device_count() > 1:
    model_transfer = nn.DataParallel(model_transfer)
    criterion_transfer = criterion_transfer
model_transfer.to(device)


# %%
def train_model(model, loader, criterion, optimizer, scheduler, n_epochs):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        correct = 0.0
        total = 0.0
        accuracy = 0.0
        test_accuracy = 0.0
        test_correct = 0.0
        test_total = 0.0
        test_loss = 0.0
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
            loss = criterion(output, target)
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
            output = model(data).squeeze()
            output = torch.unsqueeze(output, 0)
            loss = criterion(output, target)   # changes
            valid_loss += ((1 / (batch_idx + 1)) * ((loss.data) - valid_loss))
            pred = output.data.max(1, keepdim=True)[1]
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred)).cpu().numpy())) # noqa
            total += data.size(0)
        accuracy = 100. * (correct/total)
        ######################
        # Test the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loader['test']):
            # move to GPU
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            # update the average validation loss
            output = model(data).squeeze()
            output = torch.unsqueeze(output, 0)
            loss = criterion(output, target)   # changes
            valid_loss += ((1 / (batch_idx + 1)) * ((loss.data) - test_loss))
            pred = output.data.max(1, keepdim=True)[1]
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred)).cpu().numpy())) # noqa
            total += data.size(0)
        test_accuracy = 100. * (test_correct/test_total)        
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \t Validation Accuracy: {:.6f} \t Test Accuracy: {:.6f}'.format( # noqa
            epoch,
            train_loss,
            valid_loss,
            accuracy,
            test_accuracy
            ))
        scheduler.step(train_loss)
        wandb.log({'Epoch': epoch, 'loss': train_loss,'valid_loss': valid_loss, 'Valid_Accuracy': accuracy}) # noqa
        # TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format( # noqa
                valid_loss_min, valid_loss))
            # Saving the  model
            state = {
                    'optimizer': optimizer.state_dict(),
                    'state_dict':model.state_dict(),
                    }
            torch.save(model, './models/complete_model_2_224.pt')
            torch.save(model.module.state_dict(), './models/case_2_model_224.pt')
            torch.save(state, './models/case_2_state_224.pt')
            valid_loss_min = valid_loss
    return model


# %%

train_model(model=model_transfer, loader=data_transfer, optimizer=optimizer, criterion=criterion_transfer, scheduler=scheduler, n_epochs=50)
