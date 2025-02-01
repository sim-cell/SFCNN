import torch
import torch.nn as nn
import torch.nn.functional as F

class GSiLU(nn.Module):
    """Global Sigmoid Linear Unit proposed in the paper
    Returns x * sigmoid(global average pooling of x)
    """
    def __init__(self):
        super(GSiLU, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        gap = self.gap(x)
        return x * torch.sigmoid(gap)
    

class DWCONV(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DWCONV, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)

class SFCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(SFCNNBlock, self).__init__()

        self.downsample = downsample #used in Type 2 with downsample
        if downsample : stride=2 

        #1 Applying 3x3 DWConv (groups parameter separates channels, performing depthwise convolution)
        self.dwconv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        #2 Pass through PWConv and SiLU
        self.pwconv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.silu = nn.SiLU()
        #3 Applying 3x3 DWConv
        self.dwconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False)
        #4 Pass through GSILU
        self.gsilu = GSiLU()
        #5 Pass through PWConv
        self.pwconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        #7 Passing x through these at the end
        if self.downsample:
            self.layernorm = nn.LayerNorm([in_channels])
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.downsample_pw = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        input = x
        out = x

        if self.downsample:
            B, C, H, W = out.size()
            out = out.permute(0, 2, 3, 1)  # B, H, W, C (need to put C in last pos for LN)
            out = self.layernorm(out) # apply layernorm in the beginning
            out = out.permute(0, 3, 1, 2) # back to original shape

            #7 If downsample is True, apply 3x3 DWConv and PWConv to input x
            input = self.downsample_conv(x)
            input = self.downsample_pw(input)

        out = self.dwconv1(input)
        out = self.pwconv1(out)
        out = self.silu(out)

        out = self.dwconv2(out)
        out = self.gsilu(out) 
        out = self.pwconv2(out) # Output of step 5


        #6 Input of step 1 and output of step 5 are added
        out += input
        return out

class SFCNN(nn.Module):
    def __init__(self, num_classes=1000, block_numbers=[4, 8, 20, 4], channels=[48, 96, 192, 384]):
        super(SFCNN, self).__init__()

        self.stem = nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False)

        self.stage1 = self.make_stage(block_numbers[0], channels[0], channels[1], stride=2)
        self.stage2 = self.make_stage(block_numbers[1], channels[1], channels[2], stride=2)
        self.stage3 = self.make_stage(block_numbers[2], channels[2], channels[3], stride=2)
        self.stage4 = self.make_stage(block_numbers[3], channels[3], channels[3], stride=1)

        self.last_conv = nn.Conv2d(channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def make_stage(self, block_nb, in_channels, out_channels, stride):
        layers = []
        layers.append(SFCNNBlock(in_channels, out_channels, stride=stride, downsample=True))
        for _ in range(1, block_nb):
            layers.append(SFCNNBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.last_conv(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# # Test to see in there are any errors before training anything
# model = SFCNN(num_classes=1000)
# input_tensor = torch.randn(1, 3, 224, 224)
# output = model(input_tensor)
# print(output.shape)  # Result should be torch.Size([1, 1000])


# baseline (3x3 CNN)
class CNN(nn.Module):
    def __init__(self, num_classes=10, kernel_size=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=kernel_size//2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

