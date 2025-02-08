import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from modules import *

writer = SummaryWriter("runs/sfcnn_pico_cifar10")

# Hyperparamètres
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
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),  # CIFAR stats
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])

#loading the dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform) 
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform) 

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Init the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 10 #100 for cifar100


#DEFINE VERSION
print("Creating SFCNN-B")
model = SFCNN(num_classes=num_classes, block_numbers=[3, 4, 12, 3], channels=[32*2**i for i in range(4)],expand_ratio=4).to(device)
print("Executing SFCNN-B")
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)

# Warmup scheduler
def warmup_scheduler(epoch, warmup_epochs, optimizer):
    if epoch < warmup_epochs:
        lr = learning_rate * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Warmup learning rate
    warmup_scheduler(epoch, warmup_epochs, optimizer)

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        #print("starting forward pass")
        outputs = model(inputs)
        #print("outputs calculated")
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # if i % 100 == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

    # Update learning rate
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
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%')
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', val_acc, epoch)

# Save the final model
torch.save(model.state_dict(), 'models/sfcnn_pico_cifar10_run2.pth')  
writer.close()

