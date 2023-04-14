from operator import itemgetter
import os
from glob import glob
import torch
from monai.data import Dataset, CacheDataset, DataLoader
from monai.transforms import(
    Compose,
    LoadImaged,
    ToTensord,
)
from monai.utils import first

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

data_dir = "home/joshmah/eecs545_med_image/seg/data/nifti_files"
# data_dir = "data/nifti_files"
device = torch.device("cuda:0")
transforms = Compose(
        [
            LoadImaged(keys=["vol", "seg"]), # loads data from nifti files
            ToTensord(keys=["vol", "seg"]), # convert to Tensor for use in training
        ]
    ) # basic loading just for sanity check

# actual run sequence
if __name__ == '__main__':
    # load dataset
    dataset_noCache = load_data(data_dir, transforms)
    dataset_cached = load_data(data_dir, transforms, cache=True)
    # try accessing data
    print("Non cached dataset=",dataset_noCache[0]["vol"].shape)
    print("Cached dataset",dataset_cached[0]["vol"].shape)
    # insert to dataloader
    dataloader_noCache = DataLoader(dataset_noCache)
    dataloader_cached = DataLoader(dataset_cached)
    # try accessing data
    print("Non cached dataloader=",first(dataloader_noCache)["vol"].shape)
    print("Cached dataloader=",first(dataloader_cached)["vol"].shape)
    # end
    print("Completed Monai data loading test!")
