from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import DataLoader, CacheDataset, decollate_batch
from monai.utils import set_determinism
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import(
    Compose,
    LoadImaged,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    EnsureChannelFirstd,
    RandCropByPosNegLabeld,
    RandAffined,
    #RandGaussianNoised,
    RandFlipd,
    MapTransform,
    RandSpatialCropd,
    EnsureTyped,
    Activations,
    AsDiscrete,
)

import torch
import os
from glob import glob
import numpy as np

class ConvertToMultiChannelBasedOnLiverClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the liver
    label 2 is the tumor
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            #result.append(d[key] == 0)
            # merge label 1 and label 2 to construct liver
            result.append(torch.logical_or(d[key] == 1, d[key] == 2))
            # label 2 is tumor
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
    
# Data preparation and augmentation for liver and tumor segementation
def augment_data(in_dir, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, spatial_size=[128,128,64], one_state=False):
    '''
    In this function we perfrom data augmentation of the full nii files.
    The training ds is augmented with random cropping of the original data set and scale to the stipulated spatial size, 
    randAffined is then performed to rotate and shift the image

    '''

    set_determinism(seed=0)

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes_full", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainLabels_full", "*.nii.gz")))

    #path_val_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes", "*.nii.gz")))
    #path_val_segmentation = sorted(glob(os.path.join(in_dir, "TestLabels", "*.nii.gz")))

    path_val_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes_full", "*.nii.gz")))
    path_val_segmentation = sorted(glob(os.path.join(in_dir, "TestLabels_full", "*.nii.gz")))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]
    val_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_val_volumes, path_val_segmentation)]
    
    if (one_state==False):
        train_transforms = Compose([
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstd(keys="vol"),
            EnsureTyped(keys=["vol", "seg"]),
            ConvertToMultiChannelBasedOnLiverClassesd(keys="seg"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            RandSpatialCropd(keys=["vol", "seg"], roi_size=spatial_size, random_size=False),
            RandAffined(keys=["vol", "seg"], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=spatial_size, rotate_range=(0, 0, np.pi/15), scale_range=(0.1, 0.1, 0.1)),
            #RandGaussianNoised(keys=["vol"], prob=0.15, std=0.01),
            RandFlipd(keys=["vol", "seg"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["vol", "seg"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["vol", "seg"], spatial_axis=2, prob=0.5),
            ToTensord(keys=["vol", "seg"]),
            ])

        val_transforms = Compose([
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstd(keys=["vol", "seg"]),
            EnsureTyped(keys=["vol", "seg"]),
            ConvertToMultiChannelBasedOnLiverClassesd(keys="seg"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max,b_min=0.0, b_max=1.0, clip=True),            
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            ToTensord(keys=["vol", "seg"]),
            ])
    else:
        train_transforms = Compose([
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstd(keys=["vol", "seg"]),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(keys=["vol", "seg"], label_key="seg", spatial_size=spatial_size, pos=1, neg=1, num_samples=4, image_key="vol", image_threshold=0),
            RandAffined(keys=["vol", "seg"], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=spatial_size, rotate_range=(0, 0, np.pi/15), scale_range=(0.1, 0.1, 0.1)),
            #RandGaussianNoised(keys=["vol"], prob=0.15, std=0.01),
            RandFlipd(keys=["vol", "seg"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["vol", "seg"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["vol", "seg"], spatial_axis=2, prob=0.5),
            ToTensord(keys=["vol", "seg"]),
            ])

        val_transforms = Compose([
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstd(keys=["vol", "seg"]),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max,b_min=0.0, b_max=1.0, clip=True),            
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
            ToTensord(keys=["vol", "seg"]),
            ])

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0)
    val_loader = DataLoader(val_ds, batch_size=1)

    return train_loader, val_loader



# Dice metric
def dice_metric(predicted, target, one_state=False):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it 
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(to_onehot_y=one_state, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value



# Main run
if __name__ == '__main__':

    # Define 
    data_dir = '../task_data'
    model_dir = '../task_results' 
    max_epochs = 200
    val_interval = 1
    use_lr_sch = False
    one_state = False
    resunet = True

    # Augment data
    data_in = augment_data(data_dir,one_state=one_state)

    # Create resunet 
    device = torch.device("cuda:0")
    if resunet:
        num_res_units = 2
    else:
        num_res_units = 0
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=num_res_units,
        norm=Norm.BATCH,
    ).to(device)

    # Loss function DICE and ADAM optimizer
    loss = DiceLoss(to_onehot_y=one_state, sigmoid=True, squared_pred=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    
    # Do typical training 
    # Instantiate
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_val = []
    save_metric_train = []
    save_metric_val = []
    train_loader, val_loader = data_in

    # Iterate
    for epoch in range(max_epochs):
        print("-" * 20)
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        
        # Train from training ds
        for batch_data in train_loader:
            
            train_step += 1

            volume = batch_data["vol"]
            label = batch_data["seg"]
            volume, label = (volume.to(device), label.to(device))

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  
                outputs = model(volume)
                train_loss = loss(outputs, label)
            
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_epoch_loss += train_loss.item()

            train_metric = dice_metric(outputs, label, one_state=one_state)
            epoch_metric_train += train_metric
            print(
                f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
                f"Train_loss: {train_loss.item():.4f}")
        
        if use_lr_sch :
            lr_scheduler.step()
        
        train_epoch_loss /= train_step
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)
        
        epoch_metric_train /= train_step

        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)

        print(f"current epoch: {epoch + 1}, test loss: {train_epoch_loss:.4f}")

        # Perform inference with val ds
        if (epoch + 1) % val_interval == 0:

            model.eval()
            with torch.no_grad():
                val_epoch_loss = 0
                val_metric = 0
                epoch_metric_val = 0
                val_step = 0

                for val_data in val_loader:

                    val_step += 1

                    val_volume, val_label = (val_data["vol"].to(device), val_data["seg"].to(device))
                    roi_size = (128, 128, 64)
                    sw_batch_size = 1
                    with torch.cuda.amp.autocast():  
                        val_outputs = sliding_window_inference(val_volume, roi_size, sw_batch_size, model)
                    
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    val_loss = loss(val_outputs, val_label)
                    val_epoch_loss += val_loss.item()
                    val_metric = dice_metric(val_outputs, val_label, one_state=one_state)
                    epoch_metric_val += val_metric
                    
                
                val_epoch_loss /= val_step
                save_loss_val.append(val_epoch_loss)
                np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_val)

                epoch_metric_val /= val_step
                save_metric_val.append(epoch_metric_val)
                np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_val)

                # Save best metric model
                if epoch_metric_val > best_metric:
                    best_metric = epoch_metric_val
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "best_metric_model.pth"))
                
                print(
                    f"val mean dice: {epoch_metric_val:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

    # Complete training
    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")  
