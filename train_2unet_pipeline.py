from grp_transforms import *
from grp_models import *
from monai.losses import DiceLoss
from monai.data import Dataset, CacheDataset, DataLoader
from monai.data.utils import partition_dataset
import numpy as np
import torch
import os
from glob import glob
from operator import itemgetter 
from utilities import train



def load_data(data_dir, transforms, sub_folder_images="images", sub_folder_labels="labels", indices=None, cache=False):
    image_files = sorted(glob(data_dir + "/" + sub_folder_images + "/*.nii.gz")) # sorted by name to ensure match
    label_files = sorted(glob(data_dir + "/" + sub_folder_labels + "/*.nii.gz"))
    # sanity check: ensure names for both files matches
    for file_pair in zip(image_files, label_files):
        image_name, label_name = file_pair
        assert(os.path.basename(image_name) == os.path.basename(label_name))
    # create dict for volume and segmentation
    files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(image_files, label_files)]
    if indices is not None:
        files = itemgetter(*indices)(files)
    
    if cache:
        dataset = CacheDataset(data=files, transform=transforms)
    else:
        dataset = Dataset(data=files, transform=transforms)
    return dataset


data_dir = "data/nifti_files"
model_dir = "data/task_results" 
ind_file_name = "unet2_indexes"
data_indices = np.load(data_dir + '/' + ind_file_name + '.npy').tolist()

device = torch.device("cuda:0")
model = base_unet(device) # select model
transforms = unet2_baseline_transforms() # select transforms

loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True) # select loss function
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True) # select optimizer

# actual run sequence
# TODO: add in arguments so that we can push script to server and select model to train
if __name__ == '__main__':
    # load data
    dataset = load_data(data_dir, transforms, indices=data_indices, cache=True)
    # split data
    ## TODO: set determinism here
    train_dataset, test_dataset = partition_dataset(dataset, ratios=[0.8, 0.2], shuffle=True)
    # push into dataloaders to deploy to training
    data_in = (DataLoader(train_dataset), DataLoader(test_dataset))
    # training instance
    train(model, data_in, loss_function, optimizer, 600, model_dir) ## TODO: create seperate dir to store diff data