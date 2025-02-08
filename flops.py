#FLOP TESTING
# Make sure to install :
# pip install thop #see https://pypi.org/project/thop/ 

import torch
from thop import profile, clever_format
from idek import * 
from torch.utils.tensorboard import SummaryWriter

def apply_flop(model, input):
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params
num_classes = 10

config = {
            "P": [32, [3, 4, 12, 3],4],
            "N": [40, [3, 6, 17, 3],4],
            "T": [48, [4, 8, 20, 4],4],
            "S": [64, [6, 12, 28, 6],3],
            "B": [80, [8, 15, 35, 8],3]
        }

with open("flops_noexpansion.txt", "w") as f:
    f.write(f"{'Version':<10}{'Input Shape':<30}{'FLOPs':<20}{'Params (M)':<20}\n")
    f.write("="*70 + "\n")

    for version in config:
        expand_ratio = config[version][2]
        model = SFCNN(num_classes=num_classes, block_numbers=config[version][1], channels=[config[version][0]*2**i for i in range(4)]) #,expand_ratio=expand_ratio
        model.eval()
        input = torch.randn(1, 3, 224, 224)  # mock image
        macs, params = apply_flop(model, input)
        print(f"Dimensions: {input.shape}")
        print(version + f" FLOPs: {macs}")
        params = sum(param.numel() for param in model.parameters()) #like the paper
        print(version + f" Params: {params / 1e6:.2f}")
        f.write(f"{version:<10}{str(input.shape):<20}\t{macs:<20}{params:<20.2f}\n")