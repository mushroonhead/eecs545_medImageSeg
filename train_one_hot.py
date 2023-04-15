
print('#'*20)
print("Train_one_hot")
print('#'*20)

import os
import time
import argparse
import warnings

from glob import glob
import numpy as np
import torch

from monai.networks.nets import SegResNet, UNet
from monai.networks.layers import Norm
from monai.data import DataLoader, CacheDataset, decollate_batch
from monai.utils import set_determinism
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import(
    Compose,
    LoadImaged,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    EnsureChannelFirstd,
    RandFlipd,
    RandSpatialCropd,
    AsDiscrete,
    CropForegroundd,
    Resized
)

    
"""
------------------------------------------------------------------
Data agumentation and transformation
Input:
    - in_dir: data directory
    - pixdim: affine pixel dimension to scale to
    - a_min: Volume min intensity to normalize
    - a_max: Volume max intensity to normalize
    - spartial_size: ROI for crop
    - cache_rate: data loader cache, determine based on memory available
Return: 
    - labels: tensor multiclass of size (B,2,R,A,S)
------------------------------------------------------------------    
"""
def data_augment(in_dir, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, roi_size=[128,128,64], cache_rate=0.0):

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes_full", "*.nii.gz")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainLabels_full", "*.nii.gz")))

    path_val_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes_full", "*.nii.gz")))
    path_val_segmentation = sorted(glob(os.path.join(in_dir, "TestLabels_full", "*.nii.gz")))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]
    val_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_val_volumes, path_val_segmentation)]

    train_transforms = Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstd(keys=["vol", "seg"]),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
        RandSpatialCropd(keys=["vol", "seg"], roi_size=roi_size, random_size=False),
        RandFlipd(keys=["vol", "seg"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["vol", "seg"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["vol", "seg"], spatial_axis=2, prob=0.5),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["vol", "seg"]),
        ])

    val_transforms = Compose([
        LoadImaged(keys=["vol", "seg"]),
        EnsureChannelFirstd(keys=["vol", "seg"]),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max,b_min=0.0, b_max=1.0, clip=True),            
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
        Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
        Resized(keys=["vol", "seg"], spatial_size=(roi_size[0],roi_size[1],-1)),
        ToTensord(keys=["vol", "seg"]),
        ])

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=cache_rate)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=cache_rate)
    val_loader = DataLoader(val_ds, batch_size=1)

    return train_loader, val_loader



"""
------------------------------------------------------------------
Pytorch training routine with lr_scheduling
Input:
    - train_loader: training dataset after transformation
    - model: CNN model
    - criterion: Loss function
    - optimizer: Gradient solver
    - lr_scheduler: Learning rate scheduler
    - scaler: AMP scaler
    - device: device for data to
    - roi_size: expected size of train data
Return: 
    - epoch loss
------------------------------------------------------------------    
"""
def train(train_loader, model, criterion, optimizer, lr_scheduler, scaler, device, roi_size=(128,128,64)):

    model.train()
    epoch_loss = 0
    step = 0
    batch_num = 0
    
    # Train from training ds
    for batch_data in train_loader:    
        batch_num += 1
        volume = batch_data["vol"]
        label = batch_data["seg"]
    
        if ((label.size(dim=2) == roi_size[0]) and (label.size(dim=3) == roi_size[1]) and (label.size(dim=4) == roi_size[2])):
            volume, label = (volume.to(device), label.to(device))

            # Gradient solver and compute training loss
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():  
                outputs = model(volume)
                train_loss = criterion(outputs, label) 
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += train_loss.item()

            step += 1

            print(
                f"{batch_num}/{len(train_loader) // train_loader.batch_size}, "
                f"loss: {train_loss.item():.4f}")
        
        else:
            print(f"{batch_num}/{len(train_loader) // train_loader.batch_size}, label shape incorrect, skip training")
    
    lr_scheduler.step()
    epoch_loss /= step

    return epoch_loss


"""
------------------------------------------------------------------
Evaulator
Input:
    - val_loader: training dataset after transformation
    - model: CNN model
    - dice_metric: metric function
    - pre_trans: ground truth label transform
    - post_trans: output prediction transform
    - device: device for data to
Return: 
    - epoch loss
------------------------------------------------------------------
"""
def evaluate(val_loader, model, dice_metric, pre_trans, post_trans, device):
    
    model.eval()

    with torch.no_grad():
        for val_data in val_loader:
            val_volume = val_data["vol"]
            val_label = val_data["seg"]
            val_volume, val_label = (val_volume.to(device), val_label.to(device))
            
            roi_size = (128, 128, 64)
            sw_batch_size = 1

            with torch.cuda.amp.autocast():  
                val_outputs = sliding_window_inference(val_volume, roi_size, sw_batch_size, model, overlap=0.5)

            # Compute each val metrics 
            val_outputs_tf = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_label_tf = [pre_trans(i) for i in decollate_batch(val_label)]
            dice_metric(y_pred=val_outputs_tf, y=val_label_tf)  
        
        # Get mean metrics
        metric = dice_metric.aggregate().item()
        dice_metric.reset()

    return metric


"""
------------------------------------------------------------------
Main worker function
Input:
    - args
Return: 
    - Nil
------------------------------------------------------------------
"""
def main_worker(args):

    # Raise error flags
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"missing directory {args.data_dir}")
    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"missing directory {args.model_dir}")
    
    
    # Set device
    print(f"Cuda is available ? {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # Enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    
    # load dataset
    train_loader, val_loader = data_augment(args.data_dir,cache_rate=args.cache_rate)

    
    # Create net
    if args.network == 'SegResNet':
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=1,
            out_channels=3,
            dropout_prob=0.0,
            ).to(device)
        model_dir = os.path.join(args.model_dir, "segresnet_oh")
           
    elif args.network == 'ResUNet':
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256), 
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            ).to(device)
        model_dir = os.path.join(args.model_dir, "resunet_oh")

    else:            
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256), 
            strides=(2, 2, 2, 2),
            num_res_units=0,
            norm=Norm.BATCH,
            ).to(device)
        model_dir = os.path.join(args.model_dir, "unet_oh") 


    # Create loss, optimizer, lr_schedule and dice metrics
    loss_function = DiceLoss(to_onehot_y=True, 
                             sigmoid=True, 
                             squared_pred=True, 
                             smooth_nr=0, 
                             smooth_dr=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    dice_metric = DiceMetric(include_background=True, reduction="mean") 
    # Transformation into one_hot with 3 columns for 2 class
    post_trans = Compose([AsDiscrete(argmax=True, to_onehot=3)])
    pre_trans = Compose([AsDiscrete(to_onehot=3)])

    
    # Start training
    best_metric = -1
    best_metric_epoch = -1
    save_loss = []
    save_metric = []

    train_start = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print("-" * 20)
        print(f"epoch {epoch + 1}/{args.epochs}")
        epoch_loss = train(train_loader, model, loss_function, optimizer, lr_scheduler, scaler, device)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        save_loss.append(epoch_loss)
        np.save(os.path.join(model_dir, 'epoch_loss.npy'), save_loss)

        if (epoch + 1) % args.val_interval == 0:
            metric = evaluate(val_loader, model, dice_metric, pre_trans, post_trans, device)
            
            # save metrics
            save_metric.append(metric)
            np.save(os.path.join(model_dir, 'metric_mean.npy'), save_metric)

            # save model with best metric
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), 
                           os.path.join(model_dir, "best_metric_model.pth"))
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
            )
    
        print(f"time consumed for epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")

    # Train complete
    print(
    f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch},"
    f" total train time: {(time.time() - train_start):.4f}"
    )


"""
------------------------------------------------------------------
Main
------------------------------------------------------------------
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default="../task_data", type=str, help="directory of Patient CT scans dataset")
    parser.add_argument("-m", "--model_dir", default="../task_results", type=str, help="directory of train results")
    parser.add_argument("--epochs", default=500, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training.")
    parser.add_argument("--cache_rate", type=float, default=0.0, help="larger cache rate relies on enough GPU memory.")
    parser.add_argument("--val_interval", type=int, default=20)
    parser.add_argument("--network", type=str, default="SegResNet", choices=["ResUNet", "SegResNet", "UNet"])
    args = parser.parse_args()


    if args.seed is not None:
        set_determinism(seed=args.seed)
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    main_worker(args=args)


"""
------------------------------------------------------------------
Main
------------------------------------------------------------------
"""
if __name__ == '__main__':
    main()
