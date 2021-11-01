
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalization_layer(n, norm_arg):

    if n == "BN":
        return(nn.BatchNorm2d(norm_arg[0]))
    elif n == "LN":
        return(nn.LayerNorm(norm_arg))
    elif n == "GN":
        return nn.GroupNorm(norm_arg[0]//2, norm_arg[0])
    else:
        raise ValueError('Valid options are BN/LN/GN')

dropout_value = 0.005

class Net(nn.Module):
    def __init__(self, n_type='BN'):
        self.n_type = n_type
        super(Net,self).__init__()

        # Convolution Block 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),padding = 0, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [8, 26, 26])
        )

        # Convolution Block 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=10,kernel_size=(3,3),padding = 0, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [10, 24, 24])
        )

        # Transition Block 1
        self.transblock1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=10,out_channels=12,kernel_size=(3,3),padding = 0, bias = False),
            normalization_layer(self.n_type, [12, 10, 10])
        )

        # Convolution Block 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12,out_channels=14,kernel_size=(3,3),padding = 0, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [14, 8, 8])
        )

        # Convolution Block 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=14,out_channels=16,kernel_size=(3,3),padding = 0, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [16, 6, 6])
        )

        # Convolution Block 5
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size=(3,3),padding = 1, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [10, 6, 6])
        )

        # Convolution Block 6
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=12,kernel_size=(3,3),padding = 1, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [12, 6, 6])
        )

        # Convolution Block 7
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12,out_channels=14,kernel_size=(3,3),padding = 1, bias = False),
            nn.ReLU(),
            normalization_layer(self.n_type, [14, 6, 6])
        )

        # Global average pooling layer
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4),
            nn.Conv2d(in_channels=14,out_channels=10,kernel_size=(1,1),padding = 0, bias = False),
        )

    def forward(self, x):

        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.transblock1(x)

        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)

        x = self.gap(x)
        x = x.view(-1,10)

        return F.log_softmax(x,dim = -1)
