"""
To visualise the training set.

Author: Rishav
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
# from import_packages.dataset_class import Dataset
# from import_packages.dataset_partition import train_val_to_ids


def plot(train_loader, training_set):
    """To plot images."""
    fig=plt.figure(figsize=(10, 20))
    labels_dict = {1: "Excellent", 2: "Good", 3: "Adequate", 4: "Insuff_Any_All"} # noqa
    check_list = [1, 2, 3, 4]
    ax = []
    for i in range(len(train_loader)):
        image, label = training_set[i]
        if torch.is_tensor(image):
            image = image.permute(1, 2, 0)
            image = image.numpy()
            print(image.shape)
        if label in check_list:
            ax = plt.subplot(1, 4, len(check_list))
            check_list.remove(label)
    # ax.append(fig.add_subplot(rows,columns,i+1))
            ax.set_title("{}".format(labels_dict[label]))
            ax.axis('off')
            
            plt.subplots_adjust(bottom=0.7, top=0.9, hspace=0)
            plt.imshow(image.astype('uint8'))
        if not check_list:
            break


if __name__ == 'main':
    partition, labels=train_val_to_ids(csv_file='Combined_Files_without_label5.csv', stratify_columns='labels') # noqa
    training_set = Dataset(partition['train_set'], labels, root_dir='/Users/swastik/ophthalmology/Project_Quality_Assurance/Mod_AEON_data') # noqa
    train_loader = torch.utils.data.DataLoader(training_set, shuffle=True)
    plot(train_loader)
