from PIL import Image, ImageFilter
import pandas as pd 
import numpy as np
from torch.utils.data import Dataset
import os 
from torchvision import transforms

def get_paths():
    path = 'image/'
    lst_path = os.listdir(path)
    all_paths = [path+x for x in lst_path]

    return all_paths

class Data(Dataset):
  def __init__(self, paths):
    super(Data, self).__init__()
    self.paths = paths[:2000]
    
    self.trans = transforms.Compose([transforms.ToTensor()])  

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, xid):
    self.image = Image.open(self.paths[xid])
    self.image = self.image.resize((128,128))
    self.temp_img = self.image.resize((40,40), Image.BILINEAR)
    self.xs_img = self.temp_img.resize(self.image.size, Image.NEAREST)

    return self.trans(self.xs_img), self.trans(self.image)


# data = Data(get_paths())

# img, label = data[500]
# label.show()