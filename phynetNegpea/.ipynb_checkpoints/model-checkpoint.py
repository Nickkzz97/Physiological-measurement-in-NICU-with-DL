import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.start = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=32, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ELU()
        )
        
        # 1x
        self.loop1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # encoder
        self.encoder = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),

            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        #
        self.loop4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        # decoder to reach back initial temporal length
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),

            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU()
        )

        self.end = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        )

    def forward(self, x):
        #x = self.start(x)
        #print("Shape of modules in start block : ")
        for layer in self.start:
            x = layer(x)
#             print(x.size())
        
        #x = self.loop1(x)
#         print("Shape of modules in loop1 block : ")
        for layer in self.loop1:
            x = layer(x)
#             print(x.size())
#         x = self.encoder(x)
        #print("Shape of modules in encoder block : ")
        for layer in self.encoder:
            x = layer(x)
#             print(x.size())
#        x = self.loop4(x)
        #print("Shape of modules in loop4 block : ")
        for layer in self.loop4:
            x = layer(x)
#             print(x.size())
#        x = self.decoder(x)
#         print("Shape of modules in decoder block : ")
        for layer in self.decoder:
            x = layer(x)
#             print(x.size())
#         x = self.end(x)
        #print("Shape of modules in end block : ")
        for layer in self.end:
            x = layer(x)
#             print(x.size())

        return x
    
   
if __name__ == '__main__':
    
    def physnet_test():
        model = PhysNet().to('cuda')
        model.eval()
        with torch.no_grad():
            x = torch.rand(16, 3, 148, 128, 128).to('cuda')
            out =  model(x)
        print("output shape :",out.squeeze().shape)
#         #print(model)
#     physnet_test()

