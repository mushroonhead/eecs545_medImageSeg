"""
File: Group Models
Purpose: 
Stores all architecture models the group have developed
"""
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch


def base_unet(device=torch.device("cuda:0")):
    """
    Baseline UNet model the group will be improving from
    """
    return UNet(
        dimensions=3,
        in_channels=1,
        out_channels=1, # one channel output since only 1 class
        channels=(16, 32, 64, 128, 256), 
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)