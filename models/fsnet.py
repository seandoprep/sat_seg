import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn

from models.modules.fsnet_modules import *

'''
Idea from https://www.mdpi.com/2072-4292/14/5/1249
'''

# Double_convolution block
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

# U-Net main architecture
class FSNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1) -> None:
        super(FSNet, self).__init__()

        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.scblock = SCBlock(64)

        self.conv_x = nn.Conv2d(64, num_classes, 3, padding=1)
        self.conv_x_encoded = nn.Conv2d(128, num_classes, 3, padding=1)
        self.conv_y_encoded = nn.Conv2d(128, num_classes, 3, padding=1)
        self.conv_x_decoded = nn.Conv2d(32, num_classes, 3, padding=1)
        self.conv_y_decoded = nn.Conv2d(32, num_classes, 3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        conv1 = self.dconv_down1(x) 
        x = self.maxpool(conv1)  # 3,256,256 -> 64,128,128

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)  # 64,128,128 -> 128, 64, 64
        
        conv3 = self.dconv_down3(x)  # 256, 64, 64
        x = self.maxpool(conv3)  # 128, 64, 64 -> 256, 32, 32

        x = self.dconv_down4(x)  # 256, 32, 32 -> 512, 32, 32

        x = self.upsample(x)  #  512, 32, 32 -> 512, 64, 64         
        x = torch.cat([x, conv3], dim=1)  # (512 + 128), 64, 64

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   

        x = self.dconv_up1(x)

        if y != None:
            y = torch.permute(y, (0,3,1,2))
            x, x_encoded, x_decoded, y_encoded, y_decoded = self.scblock(x, y)

            pred_mask = self.conv_x(x)
            encoded_pred_mask = self.conv_x_encoded(x_encoded)
            decoded_pred_mask = self.conv_x_decoded(x_decoded)
            encoded_true_mask = self.conv_y_encoded(y_encoded)
            decoded_true_mask = self.conv_y_decoded(y_decoded)

            pred_mask = self.sigmoid(pred_mask)
            encoded_pred_mask = self.sigmoid(encoded_pred_mask)
            decoded_pred_mask = self.sigmoid(decoded_pred_mask)
            encoded_true_mask = self.sigmoid(encoded_true_mask)
            decoded_true_mask = self.sigmoid(decoded_true_mask)
            
            return pred_mask, encoded_pred_mask, decoded_pred_mask, encoded_true_mask, decoded_true_mask

        else:
            return x