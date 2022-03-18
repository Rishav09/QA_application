"""Author: Rishav Sapahia."""
# %%
import numpy as np
import os
import random
import sys
from import_packages.dataset_partition import split_equal_into_val_test
from import_packages.dataset_class import Dataset
from import_packages.train_val_to_ids import train_val_to_ids
from import_packages.checkpoint import save_checkpoint
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.utils.data.distributed
import horovod.torch as hvd
import torch.multiprocessing as mp

sys.path.insert(1, '/home/ubuntu/QA_code/')

# %%
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Horovod



def train_model(model, loader, sampler, criterion, optimizer, scheduler, n_epochs, checkpoint_path): # noqa
	"""Return trained model."""
 	# initialize tracker for minimum validation loss
	valid_loss_min = np.Inf
	for epoch in range(1, n_epochs+1):
		# initialize variables to monitor training and validation loss
		sampler['train_sampler'].set_epoch(epoch)
		train_loss = 0.0
		valid_loss = 0.0
		correct = 0.0
		total = 0.0
		accuracy = 0.0
		 # train the model #
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
			loss = criterion(output, target)   # changes
			valid_loss += ((1 / (batch_idx + 1)) * ((loss.data) - valid_loss))
			pred = torch.argmax(output, dim=1)
			correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred)).cpu().numpy())) # noqa	
			total += data.size(0)
	# Horovod
		correct /=len(sampler['val_sampler']) 
		test_correct= 100.*(correct)
		tensor = torch.tensor(correct)
		avg_tensor = hvd.allreduce(tensor, name='avg_accuracy')
		accuracy = avg_tensor.item()
		#accuracy = avg_tensor.clone().detach()	
		#loss_tensor = torch.tensor(valid_loss))
		loss_tensor = valid_loss.detach().clone()
		avg_tensor_loss = hvd.allreduce(loss_tensor, name='avg_loss')
		valid_loss = avg_tensor_loss.item()
        #accuracy = 100. * (correct/total)   
		print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \t Validation Accuracy: {:.6f} \t '.format(epoch,
		train_loss,
		valid_loss,
		accuracy,
		))
		scheduler.step(train_loss)
        #wandb.log({'Epoch': epoch, 'loss': train_loss,'valid_loss': valid_loss, 'Valid_Accuracy': accuracy}) # noqa
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
if __name__=='__main__':
	# %%
	# Deterministic Behavior
	seed = 42
	os.environ['PYTHONHASHSEED'] = str(seed)

	# Horovod
	hvd.init()

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
	temp_train,temp_valid, temp_test = split_equal_into_val_test(csv_file='/home/rxs1576/latest_scripts/Final_Input_With_normalized_labels.csv', stratify_colname='labels') # noqa
	partition, labels=train_val_to_ids(temp_train, temp_test, temp_valid, stratify_columns='labels') # noqa
	training_set = Dataset(partition['train_set'], labels, root_dir='/scratch/netra/AEON_data_600', train_transform=False) # noqa
	validation_set = Dataset(partition['val_set'],labels,root_dir='/scratch/netra/AEON_data_600',valid_transform = False) # noqa
	test_set = Dataset(partition['test_set'],labels,root_dir='/scratch/netra/AEON_data_600',test_transform=True) # noqa

	#Horovod
	cuda_avail = torch.cuda.is_available()
	kwargs = {'num_workers': 4, 'pin_memory':True} if cuda_avail else {}
	if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        	kwargs['multiprocessing_context'] = 'forkserver'


	train_sampler = torch.utils.data.distributed.DistributedSampler(training_set, num_replicas=hvd.size(), rank=hvd.rank())
	val_sampler = torch.utils.data.distributed.DistributedSampler(validation_set, num_replicas=hvd.size(), rank=hvd.rank())
	test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, num_replicas=hvd.size(), rank=hvd.rank())

	data_sampler = {
        	'train_sampler':train_sampler,
        	'val_sampler': val_sampler,
		'test_sampler': test_sampler
	}
	train_loader = torch.utils.data.DataLoader(training_set,sampler = train_sampler,batch_size=batch_size, **kwargs) # noqa
	val_loader = torch.utils.data.DataLoader(validation_set,sampler = val_sampler, batch_size=batch_size, **kwargs) # noqa
	test_loader = torch.utils.data.DataLoader(test_set, sampler = test_sampler,batch_size=batch_size, **kwargs) # noqa

	# %%
	data_transfer = {'train': train_loader,
                 'valid': val_loader,
                 'test': test_loader
		}

#model_transfer = EfficientNet.from_pretrained('efficientnet-b7')
	model_transfer = EfficientNet.from_pretrained('efficientnet-b0',weights_path='/home/rxs1576/latest_scripts/Project_QA/EfficientNetPytorch/efficientnet-b0-355c32eb.pth')
	n_inputs = model_transfer._fc.in_features
	model_transfer._fc = nn.Linear(n_inputs, 4)


	for name, parameter in model_transfer.named_parameters():
		parameter.requires_grad = False


# %%
	update_params_name = ['_fc.weight', '_fc.bias', '_conv_head.weight']
	for name, parameter in model_transfer.named_parameters():
		if name in update_params_name:
        		parameter.requires_grad = True


## Horovod
	use_adasum = True
	lr_scaler = 1 # using adasum
	if cuda_avail:
		torch.cuda.set_device(hvd.local_rank())
		model_transfer.cuda()
	if hvd.nccl_built():
		lr_scaler = hvd.local_size()
# %%    
	weights = torch.tensor([0.002, 0.0002, 0.0002, 0.0002])
	learning_rate = 3e-4
	optimizer = torch.optim.SGD(model_transfer.parameters(), lr=learning_rate*lr_scaler)
## Horovod
	hvd.broadcast_parameters(model_transfer.state_dict(), root_rank=0)
	hvd.broadcast_optimizer_state(optimizer, root_rank=0)

	optimizer = hvd.DistributedOptimizer(
		optimizer,
		named_parameters=model_transfer.named_parameters(),
		op=hvd.Adasum if use_adasum else hvd.Average,
		)

	criterion_transfer = nn.CrossEntropyLoss(weight=weights, reduction='mean')
	scheduler = ReduceLROnPlateau(
		optimizer,
		patience=4,	
		factor=0.1,

		)


	criterion_transfer.cuda()
# %%
	if torch.cuda.device_count() > 1:
#    model_transfer = nn.DataParallel(model_transfer)
		criterion_transfer = criterion_transfer

	train_model(model=model_transfer, loader=data_transfer,sampler=data_sampler,  optimizer=optimizer, criterion=criterion_transfer, scheduler=scheduler, n_epochs=3, checkpoint_path='/home/rxs1576/latest_scripts/Project_QA/checkpoint_600.pt') # noqa
