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
import tqdm
from sys import float_info


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

def train2(model, data_in, loss, optim, max_epochs, model_dir, device=torch.device("cuda:0")):
    
    def train_step(model, train_loader, optim, loss_fn, device) -> float:
        """
        Training step for each epoch
        Returns training loss normalized for given batch
        """
        train_loss = 0.
        model.train() # set model to training mode

        for data in train_loader:
            optim.zero_grad() # zero gradients
            image = data["vol"].to(device)
            label = data["seg"].to(device)
            loss = loss_fn(model(image), label)
            loss.backward() # propagate gradients
            optim.step() # iteratively update parameters
            train_loss += loss.item()

        return train_loss/len(train_loader)
    
    def val_step(model, val_loader, loss_fn, device) -> float:
        """
        Validation step for each epoch
        Returns validation loss normalized for given batch
        """
    
        val_loss = 0.
        model.eval() # set model to evaluation mode

        for data in val_loader:
            image = data["vol"].to(device)
            label = data["seg"].to(device)
            loss = loss_fn(model(image), label)
            val_loss += loss.item()

        return val_loss/len(val_loader)
    

    train_loader, test_loader = data_in
    save_loss_train = [] # stores training loss history
    save_loss_test = [] # stores testing loss history
    lowest_test_loss = float_info.max # stores the lowest test loss encountered
    best_model_epoch = -1 # the epoch iteration where best model is from

    pbar = tqdm.tqdm(range(max_epochs))
    for epoch_i in pbar:
        # train step
        train_loss_i = train_step(model, train_loader, optim, loss, device)
        # val step
        val_loss_i = val_step(model, test_loader, loss, device)
        # store traing data (in case of early termination)
        save_loss_train.append(train_loss_i)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)
        # store testing data (in case of early termination)
        save_loss_test.append(val_loss_i)
        np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)
        # store best model so far (is it right to use val data to decide?)
        if val_loss_i < lowest_test_loss:
            lowest_test_loss = val_loss_i
            best_model_epoch = epoch_i
            torch.save(model.state_dict(), os.path.join(
                        model_dir, "best_metric_model.pth")) # store first (in case of early termination)

        # set desciption output
        pbar.set_description(f'Training Loss: {train_loss_i:.4f} | Validation Loss: {val_loss_i:.4f} || Best model epoch: {best_model_epoch}')


data_dir = "data/nifti_files"
model_dir = "data/task_results" 
ind_file_name = "unet2_indexes"
data_indices = np.load(data_dir + '/' + ind_file_name + '.npy').tolist()

device = torch.device("cuda:0")
model = base_unet_w_sigmoid(device) # select model
transforms = unet2_baseline_transforms() # select transforms

# loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True) # select loss function
loss_function = DiceLoss(sigmoid=True) # select loss function
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
    train2(model, data_in, loss_function, optimizer, 1200, model_dir, device=device) ## TODO: create seperate dir to store diff data