"""
File: Group Transforms
Purpose: 
Stores all preprocessing transforms that the group have set up for easy access
"""

from monai.transforms import(
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
            Spacingd(keys=["vol", "seg"], pixdim=(1.5,1.5,1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=[128,128,64]),
            # LabelToMaskd(keys=["seg"], select_labels=[1,2]) # set labels [1, 2] to be mask ## did not work well due to non continuous data
            ThresholdIntensityd(keys=["seg"], threshold=1, above=False, cval=1), # convert 2s to 1s in seg mask
            ToTensord(keys=["vol", "seg"]),
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
            Spacingd(keys=["vol", "seg"], pixdim=(1.5,1.5,1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            Resized(keys=["vol", "seg"], spatial_size=[128,128,64]),
            MaskIntensityd(keys=["vol"], mask_key="seg"), # mask out regions that are not liver
            LabelToMaskd(keys=["seg"], select_labels=[2]), # set labels [2] to be mask
            ToTensord(keys=["vol", "seg"]),
        ]
    ) ##TODO: consider if should crop foreground after getting liver data -> for now no to make data passing between UNets easier