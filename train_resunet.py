from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import DataLoader, CacheDataset, decollate_batch
from monai.utils import set_determinism
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import(
    Compose,
    LoadImaged,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    EnsureChannelFirstd,
    RandAffined,
    RandFlipd,
    RandSpatialCropd,
    AsDiscrete,
    ToTensord,
)

import torch
import os
from glob import glob
import numpy as np
print("Instantiated")


# Data preparation and augmentation for liver and tumor segementation
def augment_data(in_dir, pixdim=(1.5, 1.5, 2.0), a_min=-200, a_max=200, spatial_size=[128,128,64]):
    '''
    In this function we perfrom data augmentation of the full nii files.
    The training ds is augmented with random cropping of the original data set and scale to the stipulated spatial size, 
    randAffined is then performed to rotate and shift the image

    '''

    set_determinism(seed=0)

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes_full", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainLabels_full", "*.nii.gz")))

    path_val_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes_full", "*.nii.gz")))
    path_val_segmentation = sorted(glob(os.path.join(in_dir, "TestLabels_full", "*.nii.gz")))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]
    val_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_val_volumes, path_val_segmentation)]

    #print(val_files)
    #print(train_files)
    
    train_transforms = Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstd(keys=["vol", "seg"]),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        CropForegroundd(keys=["vol", "seg"], source_key="vol"), 
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
        RandSpatialCropd(keys=["vol", "seg"], roi_size=spatial_size, random_size=False),
        RandAffined(keys=["vol", "seg"], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=spatial_size, rotate_range=(0, 0, np.pi/15), scale_range=(0.1, 0.1, 0.1)),
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
        CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
        ToTensord(keys=["vol", "seg"]),
        ])

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.0)   
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1)

    return train_loader, val_loader


"""
--------------------------------------------
"""
# Main run
if __name__ == '__main__':

    print("Start main")
    # Define 
    data_dir = '../task_data_half'
    model_dir = '../task_results/resunet' 
    max_epochs = 50
    val_interval = 1
    use_lr_sch = False
    resunet = True

    # Augment data
    train_loader, val_loader = augment_data(data_dir)
    print("Data loaded")

    # Create resunet 
    device = torch.device("cuda:0")
    if resunet:
        num_res_units = 2
    else:
        num_res_units = 0
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=num_res_units,
        norm=Norm.BATCH,
    ).to(device)
    print("Net created")

    # Loss function DICE and ADAM optimizer
    loss = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    dice_metric_val = DiceMetric(include_background=True,reduction="mean") 
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    print("Start training")
    
    # Do typical training 
    # Do a post transformation to a 3 classes one-hot vector for dice metric computation
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=3)])
    post_label = Compose([AsDiscrete(to_onehot=3)])
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_metric_val = []

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

            # Gradient solver and compute training loss
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  
                outputs = model(volume)
                train_loss = loss(outputs, label) 
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_epoch_loss += train_loss.item()

            #print(
            #    f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
            #    f"loss: {train_loss.item():.4f}")
        
        # Compute epoch loss 
        train_epoch_loss /= train_step
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)

        # Update learning rate scheduler
        if use_lr_sch :
            lr_scheduler.step()

        print(f"current epoch: {epoch + 1}, training loss: {train_epoch_loss:.4f}")

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

                    val_volume = val_data["vol"]
                    val_label = val_data["seg"]
                    val_volume, val_label = (val_volume.to(device), val_label.to(device))
                    
                    roi_size = (128, 128, 64)
                    sw_batch_size = 1
                    with torch.cuda.amp.autocast():  
                        val_outputs = sliding_window_inference(val_volume, roi_size, sw_batch_size, model, overlap=0.5)

                    # Compute each val metrics 
                    val_outputs_one_hot = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_label_one_hot = [post_label(i) for i in decollate_batch(val_label)]
                    dice_metric_val(y_pred=val_outputs_one_hot, y=val_label_one_hot)      
                
                # Get mean metrics
                epoch_metric_val = dice_metric_val.aggregate().item()
                save_metric_val.append(epoch_metric_val)
                np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_val)
                dice_metric_val.reset()

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
                print("-" * 20)
    # Complete training
    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")  
