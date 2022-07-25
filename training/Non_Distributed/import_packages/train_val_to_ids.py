"""Author: Rishav Sapahia."""

import pandas as pd
def train_val_to_ids(train, val, test, stratify_columns='labels'): # noqa
    """
    Convert the stratified dataset in the form of dictionary : partition['train] and labels.

    To generate the parallel code according to https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    Parameters
    -----------
    train, val, test : dataframe containing train, val, test values
    stratify_columns : The label column

    Returns
    -----------
    partition, labels:
        partition dictionary containing train and validation ids and label dictionary containing ids and their labels # noqa

    """
    train_list, val_list, test_list = train['image'].to_list(), val['image'].to_list(), test['image'].to_list() # noqa
    partition = {"train_set": train_list,
                 "val_set": val_list,
                 "test_set": test_list
                 }
    labels = dict(zip(train.image, train.labels))
    labels.update(dict(zip(val.image, val.labels)))
    labels.update(dict(zip(test.image, test.labels)))
    return partition, labels


def kfold(df, test, splits, fold, stratify_columns='labels', seed=42):
    """
    Convert train_val dataframe to folds and their ids.

    Returns
    -----------
    df_train:
        dataframe of training set.
    splits :
        splits containing the folds and the split in train_idx and val_ids
    """
    train = df
    target = train[stratify_columns].values
    train_list = (train.img.iloc[splits[fold]['train_idx']]).to_list() # It is a dataframe # noqa
    train_target = target[splits[fold]['train_idx']].tolist() # It is a numpy array # noqa

    valid_list = (train.img.iloc[splits[fold]['val_idx']]).to_list()
    valid_target = target[splits[fold]['val_idx']].tolist()

    labels = dict(zip(train_list, train_target))  # Check
    labels.update(zip(valid_list, valid_target))

    test_list = test['img'].to_list()
    labels.update(dict(zip(test_list, test.labels)))
    partition = {
        "train_set": train_list,
        "valid_set": valid_list,
        "test_set": test_list
    }
    return partition, labels

def inference_val_to_ids(csv_file=None, stratify_columns='labels'): # noqa
    """
    Convert the test set in the form of dictionary : partition['test'] and labels.

    To generate the parallel code according to https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    Parameters
    -----------
    test : dataframe containing test values
    stratify_columns : The label column

    Returns
    -----------
    partition, labels:
        partition dictionary containing test and label dictionary containing ids and their labels # noqa

    """
    test = pd.read_csv(csv_file, engine='python').iloc[:, 1:]
    test_list = test['img'].to_list() # noqa
    partition = {
                 "test_set": test_list
                 }
    labels = dict(zip(test.img, test.labels))

    return partition, labels
