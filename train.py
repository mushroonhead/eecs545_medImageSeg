from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss

import torch
from preporcess import prepare
from utilities import train

"""
Assumes user has already prepared the data according to data_preperation.ipynb, if not please start from that file
"""


"""
Step 1: defining the input and output folders

- 'in_dir' specifies the folder with all the input data to train the model from. 
If you followed data_preperation.ipynb, it should be the 'task_data' folder.
- 'model_dir' specifies the folder where all the output after training the model should go, 
including the actual model itself and any metrics that have been used to quantify the training and testing performance.

"""

data_dir = '../task_data'
model_dir = '../task_results' 

"""
Step 2: prepare the data

see preprocess.py to watch what prepare does, essentially it does the following:
- defines the preprocessing step of the image before it enters the model
- load data from 'in_dir'
- preprocess data via the above definition and loads them into pytorch dataloaders
"""
data_in = prepare(data_dir, cache=True)

"""
Step 3: define the model

- this step defines what model is being trained
- currently a basiv unet model is used for this operation
"""

device = torch.device("cuda:0")
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)


"""
Step 4: defines the lost function and optimizer used

- loss function now set to dice loss
- optimizer now set to Adam
"""

#loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

"""
Step 5: train model

Run the python script using the following command $ python train.py

see utilities.py on what train does, essentially does the following:
- defines the training procedure (epoch, batching etc)
- save data to 'model_dir'
WARNING: this step takes realllllly long (7 hours+ on laptop with good GPU RTX 3070TI)
"""

if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, 600, model_dir)

"""
Training Complete!!!

- move on to testing.ipynb to test your model
"""
