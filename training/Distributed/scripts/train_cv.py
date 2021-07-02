"""Author: Rishav Sapahia."""
import numpy as np
import os
import random

from import_packages.dataset_partition import cross_validation_train_test
from import_packages.dataset_class import Dataset
from import_packages.train_val_to_ids import kfold
import torch
from import_packages.checkpoint import save_checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

# %%
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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

os.environ['WANDB_PROJECT'] = "Project_qa"
run = wandb.init(project="Project_qa", job_type='train')

# %%
 # noqa


# %%

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
optimizer = torch.optim.SGD(model_transfer.parameters(), lr=3e-1)
criterion_transfer = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(
            optimizer,
            patience=4,
            threshold=1,
            factor=0.1,
            mode='max',
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
def train_model(model, loader, criterion, optimizer, scheduler, n_epochs, checkpoint_path, fold_no): # noqa
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
            optimizer.zero_grad()
            # move to GPU
            if torch.cuda.is_available():
                data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking = True) # noqa
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += ((1 / (batch_idx + 1)) * ((loss.data) - train_loss))
            # find the loss and update the model parameters accordingly
            # record the average training loss, using something like
            # train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))       # noqa     
        ######################
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loader['valid']):
            # move to GPU
            # bs,ncrops,c,h,w = data.size()
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            # update the average validation loss
            output = model(data)
            # output = model(data.view(-1,c,h,w))
            # output_avg = output.view(bs,ncrops,-1).mean(1)
            loss = criterion(output, target)   # changes
            valid_loss += ((1 / (batch_idx + 1) * (loss.data) - valid_loss))
            pred = output.data.max(1, keepdim=True)[1]
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred)).cpu().numpy())) # noqa
            total += data.size(0)
        # print("Restarting the scheduler:")
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,len(train_loader)) # noqa
        # for p,n in zip(model.parameters(),model._all_weights[0]):
        # if n[:6] == 'weight':
        #     print('===========\ngradient:{}\n----------\n{}'.format(n,p.grad))
        # print training/validation statistics
        accuracy = 100. * (correct/total)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format( # noqa
            epoch,
            train_loss,
            valid_loss,
            accuracy
            ))
        scheduler.step(accuracy)
        wandb.log({'Epoch': epoch, 'loss': train_loss,'valid_loss': valid_loss,'Valid_Accuracy': accuracy}) # noqa
        # TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format( # noqa
                    valid_loss_min, valid_loss))
            # Saving the  model
            state = {
                    'optimizer': optimizer.state_dict(),
                    'state_dict': model.module.state_dict(),
                    }

            save_checkpoint(state, is_best=True, checkpoint_path=None, fold=fold_no, best_model_path=checkpoint_path) # noqa
            valid_loss_min = valid_loss
    return model


batch_size = 512
df_train, df_test, splits = cross_validation_train_test(csv_file='/workdir/script/Final_Input_With_normalized_labels.csv', stratify_colname='labels') # noqa
for fold in range(5):
    print("Fold: ", fold)
    partition, labels = kfold(df_train, df_test, splits, fold, stratify_columns='labels') # noqa
    training_set = Dataset(partition['train_set'], labels, root_dir='/workdir/dataset/Mod_AEON_data_512', train_transform=True) # noqa
    validation_set = Dataset(partition['val_set'],labels,root_dir='/workdir/dataset/Mod_AEON_data_512',valid_transform=True) # noqa
    test_set = Dataset(partition['test_set'], labels, root_dir='/workdir/dataset/Mod_AEON_data_512',test_transform = None) # noqa
    train_loader = torch.utils.data.DataLoader(training_set, shuffle=True, pin_memory=True, num_workers=0, batch_size=batch_size) # noqa
    val_loader = torch.utils.data.DataLoader(validation_set, shuffle=True, pin_memory=True, num_workers=0, batch_size=batch_size) # noqa
    test_loader = torch.utils.data.DataLoader(test_set, shuffle =True, pin_memory=True, num_workers=0, batch_size=batch_size) # noqa
    data_transfer = {'train': train_loader,
                     'valid': val_loader,
                     'test': test_loader
                     }
    train_model(model=model_transfer, loader = data_transfer, optimizer = optimizer, criterion = criterion_transfer,scheduler=scheduler, n_epochs = 50,checkpoint_path='./models/checkpoint_600.pt', fold_no=fold) # noqa
