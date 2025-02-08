import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from modules import SFCNNBlock
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.segmentation import MeanIoU


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SFCNNForSegmentation(nn.Module):
    def __init__(self, num_classes=21, block_numbers=[4, 8, 20, 4], channels=[48, 96, 192, 384]):
        super().__init__()

        self.stem = nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False)

        self.stage1 = self.make_stage(block_numbers[0], channels[0], channels[1], stride=2)
        self.stage2 = self.make_stage(block_numbers[1], channels[1], channels[2], stride=2)
        self.stage3 = self.make_stage(block_numbers[2], channels[2], channels[3], stride=2)
        self.stage4 = self.make_stage(block_numbers[3], channels[3], channels[3], stride=1)

        self.last_conv = nn.Conv2d(channels[3], 1024, kernel_size=1, stride=1, padding=0, bias=False)

        # remove classifier for segmentation
        self.upsample = nn.Sequential(
        nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True),
        nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True),
        nn.Conv2d(128, num_classes, kernel_size=1)
    )

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

        x = self.upsample(x)
        return x



transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Pascal VOC dataset
train_dataset = datasets.VOCSegmentation(root="./data", year='2012', image_set='train', download=True, transform=transform, target_transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

val_dataset = datasets.VOCSegmentation(root="./data", year='2012', image_set='val', download=True, transform=transform, target_transform=transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


iou_metric = MeanIoU(num_classes=21).to(device)
model = SFCNNForSegmentation(num_classes=21).to(device) # Tiny
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.05)
scheduler = MultiStepLR(optimizer, milestones=[24, 33], gamma=0.1)

# Training
num_epochs = 36
for epoch in range(num_epochs):

    total_loss = 0
    total_miou = 0
    total_accuracy = 0
    num_batches = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.squeeze(1))
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        masks = masks.squeeze(1)

        # metrics
        iou_metric.update(preds, masks)
        total_loss += loss.item()
        num_batches += 1

    scheduler.step()

    avg_loss = total_loss / num_batches
    avg_miou = iou_metric.compute()

    iou_metric.reset()


    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, mIoU: {avg_miou:.4f}")

    # Validation
    model.eval()
    total_val_loss = 0
    num_val_batches = 0

    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device).long()

            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_masks.squeeze(1))

            val_preds = torch.argmax(val_outputs, dim=1)
            val_masks = val_masks.squeeze(1)

            iou_metric.update(val_preds, val_masks)
            total_val_loss += val_loss.item()
            num_val_batches += 1

    avg_val_loss = total_val_loss / num_val_batches
    avg_val_miou = iou_metric.compute()
    iou_metric.reset()

    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation mIoU: {avg_val_miou:.4f}")
