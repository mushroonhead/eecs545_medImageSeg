"""
File: Group Models
Purpose: 
Stores all architecture models the group have developed
"""
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import torch

class UnetWSMax(torch.nn.Module):
    def __init__(self) -> None:
        # just do something basic first
        super().__init__()
        self.model = torch.nn.Sequential(
            UNet(
                dimensions=3,
                in_channels=1,
                out_channels=1, # one channel output since only 1 class
                channels=(16, 32, 64, 128, 256), 
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
            ),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return torch.round(self.model(x))


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

def base_unet_w_sigmoid(device=torch.device("cuda:0")):
    """
    Baseline UNet model with sequential sigmoid
    """
    return UnetWSMax().to(device)