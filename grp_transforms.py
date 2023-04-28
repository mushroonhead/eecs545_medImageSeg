"""
File: Group Transforms
Purpose: 
Stores all preprocessing transforms that the group have set up for easy access
"""

import torch
from monai.config import KeysCollection
from monai.transforms import(
    Transform,
    MapTransform,
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    # Activations,
    MaskIntensityd,
    ThresholdIntensityd,
    LabelToMaskd,
)

class ForceSyncAffined(MapTransform):
    """
    Forcefully set affines of targets to source's affine
    Mainly for fixing bad data points
    """

    def __init__(self, keys: KeysCollection, source_key: str, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.source_key = source_key    

    def __call__(self, data):
        d = dict(data)
        assert self.source_key in d, f"Source key {self.source_key} not in data point."
        s_data_affine = d[self.source_key].affine
        for key in self.key_iterator(d):
            d[key].affine = s_data_affine
        return d

class Roundd(MapTransform):
    """
    Rounds target data to the closest whole number
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = torch.round(d[key])
        return d
    
class AccumulateOneHot(Transform):
    def __init__(self, source_layer : int, target_layer : int) -> None:
        super().__init__()
        self.source_layer = source_layer
        self.target_layer = target_layer

    def __call__(self, data):
        to_propagate = data[self.source_layer, ...] == 1
        data[self.target_layer, to_propagate] = data[self.source_layer, to_propagate]
        return data

def default_transforms():
    """
    Default Transforms

    These are the original set of transformations used in original pipeline
    """
    return Compose(
        [
            LoadImaged(keys=["vol", "seg"]), # loads data from nifti files
            AddChanneld(keys=["vol", "seg"]), # adds a dimension to the data
            Spacingd(keys=["vol", "seg"], pixdim=(1.5,1.5,1.0), mode=("bilinear", "nearest")), # set spacing between data to be same
            Orientationd(keys=["vol", "seg"], axcodes="RAS"), # standardize orientation of the data
            ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True), # scale intensity values to the same range for all images 
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'), # create BB around regions with data and throw away rest of data
            Resized(keys=["vol", "seg"], spatial_size=[128,128,64]), # resize the image after cropping
            ToTensord(keys=["vol", "seg"]), # convert to Tensor for use in training
        ]
    )


def unet1_baseline_transforms():
    """
    Base line transforms for UNet 1

    - Additional step of merging healthy liver and tumorous liver into same layer for segmentation
    """
    return Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            # Spacingd(keys=["vol", "seg"], pixdim=(1.5,1.5,1.0), mode=("bilinear", "nearest")), # temp remove spacing due to errant affine data
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=[128,128,64]),
            # LabelToMaskd(keys=["seg"], select_labels=[1,2]) # set labels [1, 2] to be mask ## did not work well due to non continuous data
            ThresholdIntensityd(keys=["seg"], threshold=1, above=False, cval=1), # convert 2s to 1s in seg mask
            ToTensord(keys=["vol", "seg"], dtype=torch.float32),
        ]
    )

def unet2_baseline_transforms():
    """
    Base line transforms for UNet 2

    - Crops out liver from rest of data
    - Also changes 0,1 in segmentation to 0 and 2 to 1
    """
    return Compose(
        [
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            # Spacingd(keys=["vol", "seg"], pixdim=(1.5,1.5,1.0), mode=("bilinear", "nearest")), # temp remove spacing due to errant affine data
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            MaskIntensityd(keys=["vol"], mask_key="seg"), # mask out regions that are not liver
            LabelToMaskd(keys=["seg"], select_labels=[2]), # set labels [2] to be mask
            ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=[128,128,64]),
            ToTensord(keys=["vol", "seg"], dtype=torch.float32),
        ]
    ) ##TODO: consider if should crop foreground after getting liver data -> for now no to make data passing between UNets easier