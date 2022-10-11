import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=(1 if stride==1 else 0) , stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, stride=stride)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UAE(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UAE, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64, stride=3)
        self.down1 = Down(64, 128, stride=2)
        self.down2 = Down(128, 256, stride=2)
        self.down3 = Down(256, 512)
        self.outc1 = DoubleConv(512, 128, 256)
        self.outc2 = DoubleConv(128, 64)
        self.outc3 = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.outc1(x4)
        x = self.outc2(x)
        logits = self.outc3(x)
        #print(x.size())
        return logits


class DLCPNet(UAE):
    '''
    This Class treats the case of DC
    '''

    def __init__(self):
        '''
        Initilize the Class
        '''
        n_channels = 1
        n_classes = 3
        super(DLCPNet, self).__init__(n_channels = n_channels, n_classes = n_classes)
        self.tanh = nn.Tanh()


    def forward(self, x):
        x = super(DLCPNet, self).forward(x)
        return x#self.tanh(x)
