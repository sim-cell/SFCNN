#FLOP TESTING
# Make sure to install :
# pip install thop #see https://pypi.org/project/thop/ 

import torch
from thop import profile, clever_format
from modules import * 
from torch.utils.tensorboard import SummaryWriter
from vit_pytorch import ViT


#if dimensions are 224x224
class CNN(nn.Module):
    def __init__(self, num_classes=10, kernel_size=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=kernel_size//2)
        self.fc1 = nn.Linear(128 * 56 * 56, 256)
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


def apply_flop(model, input):
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params
num_classes = 10

input_dims = 224

input = torch.randn(1, 3, input_dims, input_dims)  # mock image
model = CNN(num_classes=num_classes)
model.eval()
with open(f"results/flops/flops_cnn{input_dims}.txt", "w") as f:
    f.write(f"{'Model':<10}{'Input Shape':<30}{'FLOPs':<20}{'Params (M)':<20}\n")
    f.write("="*70 + "\n")
    macs, params = apply_flop(model, input)
    print(f"Dimensions: {input.shape}")
    print(f"CNN FLOPs: {macs}")
    print(f"CNN Params: {params}")
    f.write(f"{'CNN':<10}{str(input.shape):<30}{macs:<20}{params:<20}\n")


config = {
            "P": [32, [3, 4, 12, 3],4],
            "N": [40, [3, 6, 17, 3],4],
            "T": [48, [4, 8, 20, 4],4],
            "S": [64, [6, 12, 28, 6],3],
            "B": [80, [8, 15, 35, 8],3]
        }

with open(f"results/flops/flops_sfcnn{input_dims}.txt", "w") as f:
    f.write(f"{'Version':<10}{'Input Shape':<30}{'FLOPs':<20}{'Params (M)':<20}\n")
    f.write("="*70 + "\n")

    for version in config:
        expand_ratio = config[version][2]
        model = SFCNN(num_classes=num_classes, block_numbers=config[version][1], channels=[config[version][0]*2**i for i in range(4)]) #,expand_ratio=expand_ratio
        model.eval()
        macs, params = apply_flop(model, input)
        print(f"Dimensions: {input.shape}")
        print(version + f" FLOPs: {macs}")
        #params = sum(param.numel() for param in model.parameters()) #like the paper
        print(version + f" Params: {params}")
        f.write(f"{version:<10}{str(input.shape):<20}\t{macs:<20}{params:<20}\n")

model = ViT(
    image_size=32,
    patch_size=4,
    num_classes=num_classes,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1
)

model.eval()
input = torch.randn(1, 3, 32, 32)  

with open("results/flops/flops_vit.txt", "w") as f:
    f.write(f"{'Model':<10}{'Input Shape':<30}{'FLOPs':<20}{'Params (M)':<20}\n")
    f.write("="*70 + "\n")
    macs, params = apply_flop(model, input)
    print(f"Dimensions: {input.shape}")
    print(f"ViT FLOPs: {macs}")
    print(f"ViT Params: {params}")
    f.write(f"{'ViT':<10}{str(input.shape):<30}{macs:<20}{params:<20}\n")