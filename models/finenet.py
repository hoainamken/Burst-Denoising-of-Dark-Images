import torch
import torch.nn as nn
import math
import pdb

class FineNet(nn.Module):
    def __init__(self):
        super(FineNet, self).__init__();

        self.conv1_1 = nn.Conv2d(12, 32, kernel_size= 3, stride=1, padding=1, bias=True) 
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size= 3, stride=1, padding=1, bias=True) 
        
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True) 
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True) 
        
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True) 
        
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True) 
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True) 
        
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True) 
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True) 
        
        self.bn5 = nn.BatchNorm2d(512)
        
        self.dropout = nn.Dropout(0.5)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation = 1)

        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) 
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True) 
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True) 
        
        self.bn6 = nn.BatchNorm2d(256)

        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) 
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True) 
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True) 
        
        self.bn7 = nn.BatchNorm2d(128)

        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) 
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True) 
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True) 
        
        self.bn8 = nn.BatchNorm2d(64)
        
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) 
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True) 
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True) 
        
        self.bn9 = nn.BatchNorm2d(32)

        self.conv10 = nn.Conv2d(32, 12, kernel_size=1, stride=1, padding=0, bias=True) # 3 x 256 x 256
        
        
        
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
                
    # borrow from Learning to see in the dark pytorch version
    # https://github.com/cydonia999/Learning_to_See_in_the_Dark_PyTorch/blob/master/models/lsid.py
    def pixel_shuffle(self, input, upscale_factor, depth_first=False):
        r"""Rearranges elements in a tensor of shape :math:`[*, C*r^2, H, W]` to a
        tensor of shape :math:`[C, H*r, W*r]`.
        See :class:`~torch.nn.PixelShuffle` for details.
        Args:
            input (Tensor): Input
            upscale_factor (int): factor to increase spatial resolution by
        Examples::
            >>> ps = nn.PixelShuffle(3)
            >>> input = torch.empty(1, 9, 4, 4)
            >>> output = ps(input)
            >>> print(output.size())
            torch.Size([1, 1, 12, 12])
        """
        batch_size, channels, in_height, in_width = input.size()
        channels //= upscale_factor ** 2

        out_height = in_height * upscale_factor
        out_width = in_width * upscale_factor

        if not depth_first:
            input_view = input.contiguous().view(
                batch_size, channels, upscale_factor, upscale_factor,
                in_height, in_width)
            shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
            return shuffle_out.view(batch_size, channels, out_height, out_width)
        else:
            input_view = input.contiguous().view(batch_size, upscale_factor, upscale_factor, channels, in_height, in_width)
            shuffle_out = input_view.permute(0, 4, 1, 5, 2, 3).contiguous().view(batch_size, out_height, out_width,
                                                                                 channels)
            return shuffle_out.permute(0, 3, 1, 2)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = self.lrelu(x)        
        x = self.conv1_2(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        
        conv1 = x 
        x = self.maxpool(x) 

        x = self.conv2_1(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        
        conv2 = x 
        x = self.maxpool(x) 

        x = self.conv3_1(x)
        x = self.bn3(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.bn3(x)
        x = self.lrelu(x)
        
        conv3 = x 
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.bn4(x)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.bn4(x)
        x = self.lrelu(x)
        
        conv4 = x 
        x = self.maxpool(x)
        
        x = self.conv5_1(x) 
        x = self.bn5(x)
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x = self.bn5(x)
        x = self.lrelu(x)
        
        x = self.dropout(x)
        
        
        x = self.up6(x) 
        x = torch.cat([conv4, x], dim=1)
        
        x = self.conv6_1(x)
        x = self.bn6(x)
        x = self.lrelu(x)
        x = self.conv6_2(x)
        x = self.bn6(x)
        x = self.lrelu(x)

        x = self.up7(x) 
        x = torch.cat([conv3, x], dim=1) 
        x = self.conv7_1(x)
        x = self.bn7(x)
        x = self.lrelu(x)
        x = self.conv7_2(x)
        x = self.bn7(x)
        x = self.lrelu(x)

        x = self.up8(x) 
        x = torch.cat([conv2, x], dim=1) 
        x = self.conv8_1(x) 
        x = self.bn8(x)
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.bn8(x)
        x = self.lrelu(x)
        
        x = self.up9(x)
        x = torch.cat([conv1, x], dim=1)
        x = self.conv9_1(x) 
        x = self.bn9(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.bn9(x)
        x = self.lrelu(x)
        
        x = self.conv10(x)
        x = self.pixel_shuffle(x, 2, depth_first= True)

        return x
      