"""Author: Rishav Sapahia."""
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import pandas as pd
import time


# https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
def split_stratified_into_train_val_test(df_input, stratify_colname='y',
                                         frac_train=0.6, frac_val=0.15,
                                         frac_test=0.25,
                                         ):
    """
    Split a Pandas dataframe into three subsets (train, val, and test).

    Following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.

    """
    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' %
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' %
                         (stratify_colname))

    X = df_input   # Contains all columns.

    y = df_input[[stratify_colname]]    # Column on which to stratify

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train), # noqa
                                                          ) # noqa

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test, # noqa
                                                      ) # noqa

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test


# %%
# df_train, df_val, df_test = split_stratified_into_train_val_test(df,
#                                                                  stratify_colname='labels', # noqa
#                                                                  frac_train=0.90, # noqa
#                                                                  frac_val=0.05,   # noqa
#                                                                  frac_test=0.05)  # noqa


# %%
def split_equal_into_val_test(csv_file=None, stratify_colname='y',
                              frac_train=0.6, frac_val=0.15, frac_test=0.25,
                              no_of_classes=3
                              ):
    """
    Split a Pandas dataframe into three subsets (train, val, and test).

    Following fractional ratios provided by the user, where val and
    test set have the same number of each classes while train set have
    the remaining number of left classes
    Parameters
    ----------
    csv_file : Input data csv file to be passed
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.

    """
    df = pd.read_csv(csv_file, engine='python').iloc[:, 1:]

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' %
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df.columns:
        raise ValueError('%s is not a column in the dataframe' %
                         (stratify_colname))

    df_input = df
    # Considering that split is 90% train and rest of it is valid and test.
    sfact = int((0.1*len(df))/no_of_classes)

    # Shuffling the data frame
    df_input = df_input.sample(frac=1, random_state=42)

    # https://stackoverflow.com/questions/52279834/splitting-training-data-with-equal-number-rows-for-each-classes
    df_temp_1 = df_input[df_input['labels'] == 0][:sfact]
    df_temp_2 = df_input[df_input['labels'] == 1][:sfact]
    df_temp_3 = df_input[df_input['labels'] == 2][:sfact]
    df_temp_4 = df_input[df_input['labels'] == 3][:sfact]

    dev_test_df = pd.concat([df_temp_1, df_temp_2, df_temp_3, df_temp_4])
    dev_test_y = dev_test_df['labels']
    # Split the temp dataframe into val and test dataframes.
    df_val, df_test, dev_Y, test_Y = train_test_split(
        dev_test_df, dev_test_y,
        stratify=dev_test_y,
        test_size=0.5,
        )

    # https://stackoverflow.com/questions/39880627/in-pandas-how-to-delete-rows-from-a-data-frame-based-on-another-data-frame
    df_train = df[~df['image'].isin(dev_test_df['image'])]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)
    df_train.to_csv("/home/ubuntu/QA_code/QA_application/Train_Dev_Test_Split/df_train_{0}.csv".format(timestr)) # noqa
    df_test.to_csv("/home/ubuntu/QA_code/QA_application/Train_Dev_Test_Split/df_test_{0}.csv".format(timestr))  # noqa
    df_val.to_csv("/home/ubuntu/QA_code/QA_application/Train_Dev_Test_Split/df_val_{0}.csv".format(timestr))    # noqa
    return df_train, df_val, df_test


def cross_validation_train_test(csv_file=None, stratify_colname='labels',
                                frac_train=0.95, frac_test=0.05, n_splits=5,
                                seed=42):
    """
    Split a Pandas dataframe into two subsets (train and test).

    Following fractional ratios provided by the user, where test
    set have the same number of each classes while train set have
    the remaining number of left classes. Train set will have the
    same ratio of each class, as in the original dataframe.
    Parameters
    ----------
    csv_file : Input data csv file to be passed
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_test :
        Dataframes containing the three splits.
    Splits: Indices of the.

    """
    df = pd.read_csv(csv_file, engine='python').iloc[:, 1:]

    if frac_train + frac_test != 1.0:
        raise ValueError('fractions %f, %f do not add up to 1.0' %
                         (frac_train, frac_test))

    if stratify_colname not in df.columns:
        raise ValueError('%s is not a column in the dataframe' %
                         (stratify_colname))

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    no_of_classes = 4
    sfact = int((0.05*len(df))/no_of_classes)

    df_temp_1 = df[df['labels'] == 0][:sfact]
    df_temp_2 = df[df['labels'] == 1][:sfact]
    df_temp_3 = df[df['labels'] == 2][:sfact]
    df_temp_4 = df[df['labels'] == 3][:sfact]
    df_test = pd.concat([df_temp_1, df_temp_2, df_temp_3, df_temp_4])

    df_train = df[~df['img'].isin(df_test['img'])]

    # creating the folds
    targets = df_train[stratify_colname].values
    folds = model_selection.StratifiedKFold(n_splits=5,shuffle=True, random_state=seed) # noqa
    splits = dict()
    for fold, (train, val) in enumerate(folds.split(X=df_train, y=targets)):
        splits[fold] = dict()
        splits[fold]["train_idx"] = train
        splits[fold]["val_idx"] = val

    return df_train, df_test, splits


if __name__ == 'main':
    split_stratified_into_train_val_test(csv_file='Combined_Files_without_label5.csv', stratify_columns='labels') # noqa
    split_equal_into_val_test(csv_file='Combined_Files_without_label5.csv', stratify_columns='labels') # noqa
