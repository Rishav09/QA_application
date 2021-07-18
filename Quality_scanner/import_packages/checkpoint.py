"""
Author: Rishav Sapahia.

Functions to load and save checkpoints.
"""
import torch
import os
def save_checkpoint(state, is_best=False, checkpoint_path=None, fold=None, best_model_path=None): # noqa
    """
    state_dict: To save the models state.
    fold: To save the models in cross-validation stage.
    In case to resume training, epochs and optimizer are saved.
    is_best: Bool value
    """
    if fold:
        fold_no = "fold_"+str(fold)+"_"
        filename = os.path.basename(best_model_path)
        name, extension = filename.rsplit('.')
        new_file_name = fold_no+name+"."+extension
        torch.save(state, new_file_name)
    else:
        torch.save(state, best_model_path)


def load_checkpoint(checkpoint_path, model):
    """To Load model."""
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0' if torch.cuda.is_available() else 'cpu') # noqa
    model.load_state_dict(checkpoint['state_dict'])
    return model
