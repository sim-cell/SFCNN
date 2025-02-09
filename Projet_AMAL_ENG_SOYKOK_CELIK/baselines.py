import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from vit_pytorch import ViT

from modules import *

# Define CNN model
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


#model_name = "Baseline-CNN-Small"
model_name = "Baseline-ViT"
# TensorBoard setup
writer = SummaryWriter("runs/" + model_name)

# Hyperparameters
batch_size = 128
num_epochs = 200
learning_rate = 0.001
weight_decay = 0.05
warmup_epochs = 10

# Preprocessing and augmentation
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])

# Loading the dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform) 
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform) 

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Init the models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 10

# Models
#sfcnn_tiny = SFCNN(num_classes=num_classes, block_numbers=[4, 8, 20, 4], channels=[48, 96, 192, 384]).to(device)
#baseline_cnn_small = CNN(num_classes=num_classes, kernel_size=3).to(device)
#baseline_cnn_large = CNN(num_classes=num_classes, kernel_size=7).to(device)
baseline_vit =  ViT(
    image_size=32,
    patch_size=4,
    num_classes=num_classes,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

models = {
    #"SFCNN-Tiny": sfcnn_tiny,
    #"Baseline-CNN-Small": baseline_cnn_small,
    #"Baseline-CNN-Large": baseline_cnn_large,
    "Baseline-ViT": baseline_vit
}

criterion = nn.CrossEntropyLoss()
optimizers = {name: optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) for name, model in models.items()}
schedulers = {name: CosineAnnealingLR(opt, T_max=num_epochs - warmup_epochs) for name, opt in optimizers.items()}

# Training function
def train(model, optimizer, scheduler, model_name):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Warmup learning rate
        if epoch < warmup_epochs:
            lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        writer.add_scalar(f'Loss/Train', train_loss, epoch)
        writer.add_scalar(f'Accuracy/Train', train_acc, epoch)
        writer.add_scalar(f'Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch >= warmup_epochs:
            scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        writer.add_scalar(f'Loss/Validation', val_loss, epoch)
        writer.add_scalar(f'Accuracy/Validation', val_acc, epoch)
        print(f'[{model_name}] Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

# Train all models
for model_name, model in models.items():
    train(model, optimizers[model_name], schedulers[model_name], model_name)

# Save models
for model_name, model in models.items():
    torch.save(model.state_dict(), f'models/{model_name}.pth')

writer.close()
