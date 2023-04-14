"""
To test running a simple training on the server
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import tqdm

class SimpleNet(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4,10),
            torch.nn.ReLU(),
            torch.nn.Linear(10,1),
        )
    
    def forward(self, x):
        return self.model(x)
    
class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        for data_pt in self.data:
            yield data_pt
    
    def __getitem__(self, index) -> dict:
        return self.data[index]
    

def train_step(model, train_loader, optimizer, loss_fn, device) -> float:
    
    train_loss = 0.
    model.train() # set model to training mode

    for data in train_loader:
        optimizer.zero_grad() # zero gradients
        inp_data = data['input'].to(device)
        out_data = data['output'].to(device)
        loss = loss_fn(model(inp_data), out_data) # predict and calculate losses
        loss.backward() # propagate gradients
        optimizer.step() # iteratively update parameters
        train_loss += loss.item()

    return train_loss/len(train_loader)


def val_step(model, val_loader, loss_fn, device) -> float:

    val_loss = 0. 
    model.eval() # set model to evaluation mode

    for data in val_loader:
        inp_data = data['input'].to(device)
        out_data = data['output'].to(device)
        loss = loss_fn(model(inp_data), out_data) # predict and calculate losses
        val_loss += loss.item()

    return val_loss/len(val_loader)


def train_model(model, train_dataloader, val_dataloader, loss_fn,
  num_epochs=1000, lr=1e-3, 
  use_adams=True, w_decay=0.,
  device=torch.device("cuda:0")):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay, amsgrad=True) if use_adams else torch.optim.SGD(model.parameters(), lr=lr)

    pbar = tqdm.tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    for epoch_i in pbar:
       
        train_loss_i = train_step(model, train_dataloader, optimizer, loss_fn, device) # run train step and save train losses
        val_loss_i = val_step(model, val_dataloader, loss_fn,device) # run eval step and save losses

        pbar.set_description(f'Train Loss: {train_loss_i:.4f} | Validation Loss: {val_loss_i:.4f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

    return train_losses, val_losses


x_data = torch.arange(16, dtype=torch.float32)[:,None] + torch.arange(4, dtype=torch.float32)
y_data = x_data @ torch.arange(4, dtype=torch.float32)  + torch.randn(16, dtype=torch.float32)

dataset = SimpleDataset([{'input':x, 'output':y} for x, y in zip(x_data, y_data)])
train_data, val_data = random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_data, shuffle=True)
val_loader = DataLoader(val_data, shuffle=True)
loss_func = torch.nn.MSELoss()
model = SimpleNet().to(torch.device("cuda:0"))

train_loss, val_loss = train_model(model, train_loader, val_loader, loss_func, num_epochs=3, device=torch.device("cuda:0"))