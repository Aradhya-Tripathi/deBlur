from models import Encoder, Decoder, Model
from tqdm import tqdm
from data import Data, get_paths
import torch 
from torch import nn as nn 
from torch.nn import functional as f 
from PIL import Image

encoder = Encoder(in_channel=3, out_channel=32)
decoder = Decoder(in_channel=96, out_channel=128)
model = Model(encoder, decoder)

loss_fn = nn.L1Loss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)

data = Data(get_paths())
train_dl = torch.utils.data.DataLoader(data)

def train(model=model, train_dl=train_dl, loss_fn=loss_fn, optim=optim):
    epoch_loss = 0
    model.train()

    for x, y in tqdm(train_dl, total=len(train_dl), leave=False):
        optim.zero_grad()

        model_out = model(x)
        loss = loss_fn(model_out, y)
        loss.backward()

        optim.step()

        epoch_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    return epoch_loss/len(train_dl)

for epoch in range(50):
    loss = train()
    print(f'LOSS: {loss} EPOCH: {epoch}')

    if epoch%10==0:
        torch.save(model.state_dict(), f'MODEL_{epoch}.pt')
