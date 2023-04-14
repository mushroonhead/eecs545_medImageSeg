from monai.networks.nets import SegResNetVAE
from monai.data import DataLoader, CacheDataset
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
    EnsureChannelFirstd,
    RandAffined,
    RandFlipd,
    MapTransform,
    RandSpatialCropd,
    EnsureTyped,
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
    The possible classes are liver and tumor

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
    
# Data preparation
def data_augment(in_dir, pixdim=(1.5, 1.5, 1.0), a_min=-200, a_max=200, spatial_size=[128,128,64]):

    set_determinism(seed=0)

    path_train_volumes = sorted(glob(os.path.join(in_dir, "TrainVolumes_full", "*.nii")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "TrainLabels_full", "*.nii")))

    path_val_volumes = sorted(glob(os.path.join(in_dir, "TestVolumes_full", "*.nii")))
    path_val_segmentation = sorted(glob(os.path.join(in_dir, "TestLabels_full", "*.nii")))

    train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]
    val_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_val_volumes, path_val_segmentation)]

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

    test_transforms = Compose([
            LoadImaged(keys=["vol", "seg"]),
            EnsureChannelFirstd(keys="vol"),
            EnsureTyped(keys=["vol", "seg"]),
            ConvertToMultiChannelBasedOnLiverClassesd(keys="seg"),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max,b_min=0.0, b_max=1.0, clip=True), 
            ToTensord(keys=["vol", "seg"]),
        ])

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    val_ds = CacheDataset(data=val_files, transform=test_transforms, cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1)

    return train_loader, val_loader

# Dice metric
def dice_metric(predicted, target):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it 
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value

if __name__ == '__main__':
    
    # Define 
    data_dir = '../task_data'
    model_dir = '../task_results' 
    max_epochs = 300
    test_interval = 1
    input_image_size = (128,128,64)

    data_in = data_augment(data_dir, spatial_size=input_image_size)

    # Create net    
    device = torch.device("cuda:0")

    model = SegResNetVAE(
        input_image_size=input_image_size,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=1,
        out_channels=2,
        dropout_prob=0.2,
    ).to(device)


    # Loss function and optimizer
    loss = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-5)

    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    # Do typical training 
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []
    train_loader, test_loader = data_in

    for epoch in range(max_epochs):
        print("-" * 20)
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        for batch_data in train_loader:
            
            train_step += 1

            volume = batch_data["vol"]
            label = batch_data["seg"]
            volume, label = (volume.to(device), label.to(device))

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(): 
                outputs, hidden = model(volume)
                train_loss = loss(outputs, label)
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_epoch_loss += train_loss.item()

            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
        
        lr_scheduler.step()
        
        train_epoch_loss /= train_step
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)
        
        epoch_metric_train /= train_step

        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)

        if (epoch + 1) % test_interval == 0:

            model.eval()
            with torch.no_grad():
                test_epoch_loss = 0
                test_metric = 0
                epoch_metric_test = 0
                test_step = 0

                for test_data in test_loader:

                    test_step += 1

                    test_volume = test_data["vol"]
                    test_label = test_data["seg"]
                    test_volume, test_label = (test_volume.to(device), test_label.to(device),)
                    
                    roi_size = input_image_size
                    sw_batch_size = 1
                    with torch.cuda.amp.autocast(): 
                        test_outputs, _ = model(test_volume)
                    
                    test_loss = loss(test_outputs, test_label)
                    test_epoch_loss += test_loss.item()
                    test_metric = dice_metric(test_outputs, test_label)
                    epoch_metric_test += test_metric
                    
                
                test_epoch_loss /= test_step
                #print(f'test_loss_epoch: {test_epoch_loss:.4f}')
                save_loss_test.append(test_epoch_loss)
                np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)

                epoch_metric_test /= test_step
                #print(f'test_dice_epoch: {epoch_metric_test:.4f}')
                save_metric_test.append(epoch_metric_test)
                np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)

                if epoch_metric_test > best_metric:
                    best_metric = epoch_metric_test
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "best_metric_model.pth"))
                
                print(
                    f"current epoch: {epoch + 1}, test loss: {train_epoch_loss:.4f}, val mean dice: {epoch_metric_test:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )


    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")

