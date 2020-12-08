import torch 
from torch import nn 
import torchvision 
from torch.nn import functional as f 

class Encoder(nn.Module):
  def __init__(self, in_channel, out_channel):
    super(Encoder, self).__init__()
    self.encode = nn.Sequential(
                                    nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                                    nn.MaxPool2d((2,2)),
                                    nn.LeakyReLU(),
                                    nn.Upsample(scale_factor=2),
                                    nn.Dropout(),
                                    nn.Conv2d(out_channel, out_channel*2, kernel_size=3, padding=1),
                                    nn.MaxPool2d((2,2)),
                                    nn.LeakyReLU(),
                                    nn.Upsample(scale_factor=2),
                                    nn.Dropout(),
                                    nn.Conv2d(out_channel*2, out_channel*3, kernel_size=3, padding=1),
                                    nn.MaxPool2d((2,2)),
                                    nn.LeakyReLU(),
                                    nn.Dropout(),
                                    )
  def forward(self, image):
      return self.encode(image)


class Decoder(nn.Module):
  def __init__(self, in_channel, out_channel, final_out):
    super(Decoder, self).__init__()
    self.deconv = torch.nn.Sequential(
                                            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, padding=1),
                                            nn.Dropout(p=0.5),
                                            nn.LeakyReLU(),
                                            nn.Upsample(scale_factor=2),
                                            nn.ConvTranspose2d(out_channel, 64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.LeakyReLU(),
                                            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(32),
                                            nn.ConvTranspose2d(32, final_out, kernel_size=3, padding=1),

                                        )

  def forward(self, image):
    return self.deconv(image)                                        

class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image):
        enc_img = self.encoder(image)
        return self.decoder(enc_img)

