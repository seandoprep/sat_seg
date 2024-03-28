import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import copy
import torch
import torch.nn as nn

from torch.nn import (Conv2d, Module, ModuleList, PReLU,
                      Sequential, BatchNorm2d, ReLU)
from typing import Any

''' 
Modules for FSNet
'''
##################################################################################################
#                                      Fourier Block                                             #
##################################################################################################

class Spectra(Module):
    def __init__(self,in_depth,AF='prelu'):
        super().__init__()
        
        #Params
        self.in_depth = in_depth
        self.inter_depth = self.in_depth//2 if in_depth>=2 else self.in_depth

        #Layers
        self.AF1 = ReLU if AF=='relu' else PReLU(self.inter_depth)
        self.AF2 = ReLU if AF=='relu' else PReLU(self.inter_depth)
        self.inConv = Sequential(Conv2d(self.in_depth,self.inter_depth,1),
                                    BatchNorm2d(self.inter_depth),
                                    self.AF1)
        self.midConv = Sequential(Conv2d(self.inter_depth,self.inter_depth,1),
                                    BatchNorm2d(self.inter_depth),
                                    self.AF2)
        self.outConv = Conv2d(self.inter_depth, self.in_depth, 1)
        
    def forward(self,x):
        x = self.inConv(x)
        skip = copy.copy(x)
        rfft = torch.fft.rfft2(x)  
        real_rfft = torch.real(rfft)  
        imag_rfft = torch.imag(rfft)  
        cat_rfft = torch.cat((real_rfft,imag_rfft),dim=-1)  
        cat_rfft = self.midConv(cat_rfft)
        mid = cat_rfft.shape[-1]//2  
        real_rfft = cat_rfft[...,:mid]
        imag_rfft = cat_rfft[...,mid:]
        rfft = torch.complex(real_rfft,imag_rfft)  #
        spect = torch.fft.irfft2(rfft)
        out = self.outConv(spect + skip)  
        return out
    

class FastFC(Module):
    def __init__(self,in_depth,out_depth,AF='prelu'):
        super().__init__()
        #Params
        self.in_depth = in_depth//2
        self.out_depth = out_depth
        
        #Layers
        self.AF1 = ReLU if AF=='relu' else PReLU(self.in_depth)
        self.AF2 = ReLU if AF=='relu' else PReLU(self.in_depth)
        self.conv_ll = Conv2d(self.in_depth,self.in_depth,3,padding='same')
        self.conv_lg = Conv2d(self.in_depth,self.in_depth,3,padding='same')
        self.conv_gl = Conv2d(self.in_depth,self.in_depth,3,padding='same')
        self.conv_gg = Spectra(self.in_depth, AF)
        self.bnaf1 = Sequential(BatchNorm2d(self.in_depth),self.AF1)
        self.bnaf2 = Sequential(BatchNorm2d(self.in_depth),self.AF2)
        self.conv_final = Conv2d(self.in_depth*2,self.out_depth,3,padding='same')
        
    def forward(self,x):
        #print('x shape : ', str(x.shape))
        
        mid = x.shape[1]//2
        x_loc = x[:,:mid,:,:]
        if x.shape[1]%2 != 0:
            x_glo = x[:,mid+1:,:,:]
        else:
            x_glo = x[:,mid:,:,:]
        
        #print('x_loc shape : ', str(x_loc.shape))
        #print('x_glo shape : ', str(x_glo.shape))

        x_ll = self.conv_ll(x_loc)
        x_lg = self.conv_lg(x_loc)
        x_gl = self.conv_gl(x_glo)
        x_gg = self.conv_gg(x_glo)
        out_loc = torch.add((self.bnaf1(x_ll + x_gl)),x_loc)
        out_glo = torch.add((self.bnaf2(x_gg + x_lg)),x_glo)
        out = torch.cat((out_loc,out_glo),dim=1)
        out = self.conv_final(out)
        return out,out_loc,out_glo

    
class FourierBlock(Module):
    def __init__(self,num_layer,in_depth,out_depth,return_all=False, attention=False):
        super().__init__()
        #Params
        self.num_layers = num_layer
        self.in_depth = in_depth
        self.out_depth = out_depth
        self.return_all  = return_all
        #Layers
        self.block = ModuleList()
        for _ in range(self.num_layers):
            self.block.append(FastFC(self.in_depth,self.out_depth,'prelu'))

    def forward(self,x):  
        for layer in self.block:
            x,x_loc,x_glo = layer(x)
        if self.return_all:
            return x,x_loc,x_glo
        else:
            return x
    

##################################################################################################
#                                      Shape Contraint Block                                     #
##################################################################################################

# fourier_convolution block
def ffc_conv(in_channels, out_channels):
    return nn.Sequential(
        FourierBlock(2, in_channels, out_channels),
        nn.ReLU(inplace=True),
    )


class SCBlock(Module):
    def __init__(self, in_channel):
        super().__init__()

        self.ffc_encode_pred = Conv2d(in_channel, 64, 3, padding='same')
        self.ffc_encode_true = Conv2d(1, 64, 3, padding='same')

        self.encode = nn.Sequential(
            ffc_conv(64, 64),
            Conv2d(64, 128, 3, padding='same'),
            ffc_conv(128, 128)
        )

        self.ffc_inter = ffc_conv(128, 128)
        
        self.decode = nn.Sequential(
            ffc_conv(128, 128),
            Conv2d(128, 64, 3, padding='same'),
            ffc_conv(64, 64),
            Conv2d(64, 32, 3, padding='same'),
        )

    def forward(self, x, y):  
        
        # Pred mask
        x_pred = self.ffc_encode_pred(x)
        x_encoded = self.encode(x_pred)
        x_inter = self.ffc_inter(x_encoded)
        x_decoded = self.decode(x_inter)

        # True mask
        y = self.ffc_encode_true(y)
        y_encoded = self.encode(y)
        y_inter = self.ffc_inter(y_encoded)
        y_decoded = self.decode(y_inter)

        return x, x_encoded, x_decoded, y_encoded, y_decoded