import torch
import torch.nn as nn
import math

class CoarseNet(nn.Module):
    def __init__(self):
        super(CoarseNet, self).__init__();

        self.conv1_1 = nn.Conv2d(4, 32, kernel_size= 3, stride=1, padding=1, bias=True) 
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size= 3, stride=1, padding=1, bias=True)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True) 
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True) 

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation = 1)

        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) 
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)

        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) 
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True) 
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True) 

        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) 
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True) 
        
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9_1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True) 
        self.conv9_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True) 
        

        self.conv10 = nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0, bias=True) 

        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                stdv = 1. / math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv) 
                if m.bias is not None:
                  m.bias.data.uniform_(-stdv, stdv) 

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        # encode 
        x = self.conv1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        
        conv1 = x 
        x = self.maxpool(x) 

        x = self.conv2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        
        conv2 = x 
        x = self.maxpool(x) 

        x = self.conv3_1(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        
        conv3 = x
        x = self.maxpool(x)

        x = self.conv4_1(x)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        
        conv4 = x 
        x = self.maxpool(x)
        
        x = self.conv5_1(x) 
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x = self.lrelu(x)
        
        
        x = self.up6(x)
        x = torch.cat([conv4, x], dim=1)

        x = self.conv6_1(x)
        x = self.conv6_2(x)

        x = self.up7(x)
        x = torch.cat([conv3, x], dim=1) 
        x = self.conv7_1(x)
        x = self.lrelu(x)
        x = self.conv7_2(x)
        x = self.lrelu(x)

        x = self.up8(x)
        x = torch.cat([conv2, x], dim=1) 
        x = self.conv8_1(x) 
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.lrelu(x)
        
        x = self.up9(x)
        x = torch.cat([conv1, x], dim=1)
        x = self.conv9_1(x) 
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.lrelu(x)
        
        x = self.conv10(x) 

        return x
      