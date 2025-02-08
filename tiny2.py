import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from idek import *

writer = SummaryWriter("runs/sfcnn_tiny_cifar10_noexpansion")

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


#DEFINING THE TINY VERSION
print("Creating SFCNN-T")
model = SFCNN(num_classes=num_classes, block_numbers=[4, 8, 20, 4], channels=[48, 96, 192, 384]).to(device)
print("Executing SFCNN-T")
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
torch.save(model.state_dict(), 'models/sfcnn_tiny_cifar10_run2.pth')  
writer.close()


#res
# Files already downloaded and verified
# Files already downloaded and verified
# Epoch [1/200], Step [1/391], Loss: 2.3021, Acc: 12.50%
# Epoch [1/200], Step [101/391], Loss: 1.9875, Acc: 22.04%
# Epoch [1/200], Step [201/391], Loss: 2.0154, Acc: 25.31%
# Epoch [1/200], Step [301/391], Loss: 1.8041, Acc: 27.25%

# Epoch [1/200], Validation Loss: 1.8580, Validation Acc: 35.51%
# Epoch [2/200], Step [1/391], Loss: 1.7579, Acc: 39.84%
# Epoch [2/200], Step [101/391], Loss: 1.8987, Acc: 31.65%
# Epoch [2/200], Step [201/391], Loss: 1.7874, Acc: 32.26%
# Epoch [2/200], Step [301/391], Loss: 1.8715, Acc: 32.53%
# Epoch [2/200], Validation Loss: 1.8169, Validation Acc: 36.38%
# Epoch [3/200], Step [1/391], Loss: 1.9651, Acc: 28.91%
# Epoch [3/200], Step [101/391], Loss: 1.8897, Acc: 33.07%
# Epoch [3/200], Step [201/391], Loss: 1.8266, Acc: 34.42%
# Epoch [3/200], Step [301/391], Loss: 1.7138, Acc: 34.57%
# Epoch [3/200], Validation Loss: 1.7133, Validation Acc: 39.04%
# Epoch [4/200], Step [1/391], Loss: 1.7715, Acc: 35.16%
# Epoch [4/200], Step [101/391], Loss: 1.6915, Acc: 36.80%
# Epoch [4/200], Step [201/391], Loss: 1.6813, Acc: 37.28%
# Epoch [4/200], Step [301/391], Loss: 1.7287, Acc: 37.90%
# Epoch [4/200], Validation Loss: 1.5918, Validation Acc: 43.17%
# Epoch [5/200], Step [1/391], Loss: 1.7653, Acc: 35.94%
# Epoch [5/200], Step [101/391], Loss: 1.6477, Acc: 40.53%
# Epoch [5/200], Step [201/391], Loss: 1.5367, Acc: 41.15%
# Epoch [5/200], Step [301/391], Loss: 1.7113, Acc: 41.41%
# Epoch [5/200], Validation Loss: 1.5076, Validation Acc: 46.64%
# Epoch [6/200], Step [1/391], Loss: 1.7761, Acc: 37.50%
# Epoch [6/200], Step [101/391], Loss: 1.6935, Acc: 42.86%
# Epoch [6/200], Step [201/391], Loss: 1.2438, Acc: 43.03%
# Epoch [6/200], Step [301/391], Loss: 1.6176, Acc: 43.29%
# Epoch [6/200], Validation Loss: 1.4699, Validation Acc: 46.00%
# Epoch [7/200], Step [1/391], Loss: 1.4939, Acc: 48.44%
# Epoch [7/200], Step [101/391], Loss: 1.4682, Acc: 44.07%
# Epoch [7/200], Step [201/391], Loss: 1.5322, Acc: 44.57%
# Epoch [7/200], Step [301/391], Loss: 1.4431, Acc: 45.11%
# Epoch [7/200], Validation Loss: 1.3778, Validation Acc: 50.30%
# Epoch [8/200], Step [1/391], Loss: 1.4187, Acc: 52.34%
# Epoch [8/200], Step [101/391], Loss: 1.4719, Acc: 46.96%
# Epoch [8/200], Step [201/391], Loss: 1.4512, Acc: 47.38%
# Epoch [8/200], Step [301/391], Loss: 1.4166, Acc: 47.75%
# Epoch [8/200], Validation Loss: 1.3379, Validation Acc: 51.73%
# Epoch [9/200], Step [1/391], Loss: 1.1888, Acc: 52.34%
# Epoch [9/200], Step [101/391], Loss: 1.5498, Acc: 48.04%
# Epoch [9/200], Step [201/391], Loss: 1.3626, Acc: 48.65%
# Epoch [9/200], Step [301/391], Loss: 1.3558, Acc: 49.26%
# Epoch [9/200], Validation Loss: 1.2977, Validation Acc: 54.11%
# Epoch [10/200], Step [1/391], Loss: 1.3600, Acc: 43.75%
# Epoch [10/200], Step [101/391], Loss: 1.4496, Acc: 50.94%
# Epoch [10/200], Step [201/391], Loss: 1.3123, Acc: 50.68%
# Epoch [10/200], Step [301/391], Loss: 1.2842, Acc: 50.89%
# Epoch [10/200], Validation Loss: 1.2526, Validation Acc: 55.32%
# Epoch [11/200], Step [1/391], Loss: 1.2325, Acc: 57.03%
# Epoch [11/200], Step [101/391], Loss: 1.1576, Acc: 52.94%
# Epoch [11/200], Step [201/391], Loss: 1.3847, Acc: 53.25%
# Epoch [11/200], Step [301/391], Loss: 1.2707, Acc: 53.63%
# Epoch [11/200], Validation Loss: 1.1830, Validation Acc: 57.77%
# Epoch [12/200], Step [1/391], Loss: 1.2898, Acc: 50.78%
# Epoch [12/200], Step [101/391], Loss: 1.2951, Acc: 55.39%
# Epoch [12/200], Step [201/391], Loss: 1.1167, Acc: 55.63%
# Epoch [12/200], Step [301/391], Loss: 1.1094, Acc: 55.74%
# Epoch [12/200], Validation Loss: 1.1284, Validation Acc: 59.94%
# Epoch [13/200], Step [1/391], Loss: 1.1500, Acc: 60.16%
# Epoch [13/200], Step [101/391], Loss: 1.3561, Acc: 57.19%
# Epoch [13/200], Step [201/391], Loss: 1.0714, Acc: 58.03%
# Epoch [13/200], Step [301/391], Loss: 0.9645, Acc: 58.31%
# Epoch [13/200], Validation Loss: 1.0613, Validation Acc: 62.25%
# Epoch [14/200], Step [1/391], Loss: 1.0716, Acc: 61.72%
# Epoch [14/200], Step [101/391], Loss: 1.1451, Acc: 60.02%
# Epoch [14/200], Step [201/391], Loss: 0.9804, Acc: 60.11%
# Epoch [14/200], Step [301/391], Loss: 1.0093, Acc: 60.25%
# Epoch [14/200], Validation Loss: 1.0330, Validation Acc: 63.41%
# Epoch [15/200], Step [1/391], Loss: 1.0417, Acc: 65.62%
# Epoch [15/200], Step [101/391], Loss: 0.9925, Acc: 61.03%
# Epoch [15/200], Step [201/391], Loss: 1.1563, Acc: 61.38%
# Epoch [15/200], Step [301/391], Loss: 1.0298, Acc: 61.57%
# Epoch [15/200], Validation Loss: 1.0164, Validation Acc: 63.89%
# Epoch [16/200], Step [1/391], Loss: 1.0665, Acc: 64.84%
# Epoch [16/200], Step [101/391], Loss: 0.8366, Acc: 63.10%
# Epoch [16/200], Step [201/391], Loss: 1.1615, Acc: 62.46%
# Epoch [16/200], Step [301/391], Loss: 1.0440, Acc: 62.40%
# Epoch [16/200], Validation Loss: 0.9603, Validation Acc: 65.81%
# Epoch [17/200], Step [1/391], Loss: 0.9652, Acc: 67.19%
# Epoch [17/200], Step [101/391], Loss: 0.8603, Acc: 64.46%
# Epoch [17/200], Step [201/391], Loss: 1.1461, Acc: 63.77%
# Epoch [17/200], Step [301/391], Loss: 1.0058, Acc: 64.04%
# Epoch [17/200], Validation Loss: 0.9674, Validation Acc: 65.91%
# Epoch [18/200], Step [1/391], Loss: 0.9579, Acc: 64.06%
# Epoch [18/200], Step [101/391], Loss: 0.8923, Acc: 65.06%
# Epoch [18/200], Step [201/391], Loss: 1.0774, Acc: 64.83%
# Epoch [18/200], Step [301/391], Loss: 0.7882, Acc: 64.95%
# Epoch [18/200], Validation Loss: 0.9254, Validation Acc: 67.55%
# Epoch [19/200], Step [1/391], Loss: 0.9129, Acc: 67.19%
# Epoch [19/200], Step [101/391], Loss: 1.0540, Acc: 66.27%
# Epoch [19/200], Step [201/391], Loss: 0.9673, Acc: 65.97%
# Epoch [19/200], Step [301/391], Loss: 1.1455, Acc: 66.30%
# Epoch [19/200], Validation Loss: 0.9083, Validation Acc: 67.63%
# Epoch [20/200], Step [1/391], Loss: 0.8979, Acc: 70.31%
# Epoch [20/200], Step [101/391], Loss: 0.8282, Acc: 67.79%
# Epoch [20/200], Step [201/391], Loss: 0.7672, Acc: 67.49%
# Epoch [20/200], Step [301/391], Loss: 0.8555, Acc: 67.10%
# Epoch [20/200], Validation Loss: 0.9366, Validation Acc: 67.05%
# Epoch [21/200], Step [1/391], Loss: 0.8314, Acc: 73.44%
# Epoch [21/200], Step [101/391], Loss: 0.8741, Acc: 67.79%
# Epoch [21/200], Step [201/391], Loss: 0.9350, Acc: 67.49%
# Epoch [21/200], Step [301/391], Loss: 0.9243, Acc: 67.52%
# Epoch [21/200], Validation Loss: 0.8713, Validation Acc: 70.07%
# Epoch [22/200], Step [1/391], Loss: 0.9357, Acc: 63.28%
# Epoch [22/200], Step [101/391], Loss: 0.9246, Acc: 68.60%
# Epoch [22/200], Step [201/391], Loss: 1.0000, Acc: 68.80%
# Epoch [22/200], Step [301/391], Loss: 0.8204, Acc: 68.86%
# Epoch [22/200], Validation Loss: 0.8479, Validation Acc: 70.72%
# Epoch [23/200], Step [1/391], Loss: 0.7673, Acc: 68.75%
# Epoch [23/200], Step [101/391], Loss: 0.9628, Acc: 69.74%
# Epoch [23/200], Step [201/391], Loss: 0.7252, Acc: 69.54%
# Epoch [23/200], Step [301/391], Loss: 0.8379, Acc: 69.48%
# Epoch [23/200], Validation Loss: 0.8408, Validation Acc: 69.90%
# Epoch [24/200], Step [1/391], Loss: 0.9085, Acc: 67.19%
# Epoch [24/200], Step [101/391], Loss: 0.7683, Acc: 70.68%
# Epoch [24/200], Step [201/391], Loss: 0.6478, Acc: 70.90%
# Epoch [24/200], Step [301/391], Loss: 1.0219, Acc: 70.44%
# Epoch [24/200], Validation Loss: 0.8581, Validation Acc: 70.11%
# Epoch [25/200], Step [1/391], Loss: 0.6353, Acc: 76.56%
# Epoch [25/200], Step [101/391], Loss: 0.9174, Acc: 70.96%
# Epoch [25/200], Step [201/391], Loss: 0.8569, Acc: 70.82%
# Epoch [25/200], Step [301/391], Loss: 0.9903, Acc: 70.98%
# Epoch [25/200], Validation Loss: 0.8210, Validation Acc: 71.54%
# Epoch [26/200], Step [1/391], Loss: 0.7338, Acc: 71.09%
# Epoch [26/200], Step [101/391], Loss: 0.6381, Acc: 71.13%
# Epoch [26/200], Step [201/391], Loss: 0.7877, Acc: 71.21%
# Epoch [26/200], Step [301/391], Loss: 0.9307, Acc: 71.45%
# Epoch [26/200], Validation Loss: 0.7881, Validation Acc: 72.54%
# Epoch [27/200], Step [1/391], Loss: 0.7876, Acc: 71.09%
# Epoch [27/200], Step [101/391], Loss: 0.8678, Acc: 72.67%
# Epoch [27/200], Step [201/391], Loss: 0.8406, Acc: 72.33%
# Epoch [27/200], Step [301/391], Loss: 0.7074, Acc: 72.11%
# Epoch [27/200], Validation Loss: 0.7694, Validation Acc: 73.07%
# Epoch [28/200], Step [1/391], Loss: 0.8153, Acc: 67.97%
# Epoch [28/200], Step [101/391], Loss: 0.9079, Acc: 72.48%
# Epoch [28/200], Step [201/391], Loss: 0.9408, Acc: 72.56%
# Epoch [28/200], Step [301/391], Loss: 0.7010, Acc: 72.68%
# Epoch [28/200], Validation Loss: 0.7620, Validation Acc: 73.08%
# Epoch [29/200], Step [1/391], Loss: 0.7073, Acc: 73.44%
# Epoch [29/200], Step [101/391], Loss: 0.6724, Acc: 73.12%
# Epoch [29/200], Step [201/391], Loss: 0.8603, Acc: 73.14%
# Epoch [29/200], Step [301/391], Loss: 0.6753, Acc: 72.93%
# Epoch [29/200], Validation Loss: 0.7563, Validation Acc: 73.57%
# Epoch [30/200], Step [1/391], Loss: 0.6900, Acc: 76.56%
# Epoch [30/200], Step [101/391], Loss: 0.8214, Acc: 73.96%
# Epoch [30/200], Step [201/391], Loss: 0.7872, Acc: 73.95%
# Epoch [30/200], Step [301/391], Loss: 0.7016, Acc: 73.73%
# Epoch [30/200], Validation Loss: 0.7500, Validation Acc: 73.44%
# Epoch [31/200], Step [1/391], Loss: 0.7360, Acc: 74.22%
# Epoch [31/200], Step [101/391], Loss: 0.8001, Acc: 74.15%
# Epoch [31/200], Step [201/391], Loss: 0.7325, Acc: 74.26%
# Epoch [31/200], Step [301/391], Loss: 0.6578, Acc: 74.04%
# Epoch [31/200], Validation Loss: 0.7149, Validation Acc: 75.31%
# Epoch [32/200], Step [1/391], Loss: 0.6424, Acc: 75.00%
# Epoch [32/200], Step [101/391], Loss: 0.8768, Acc: 74.74%
# Epoch [32/200], Step [201/391], Loss: 0.8628, Acc: 74.30%
# Epoch [32/200], Step [301/391], Loss: 0.7139, Acc: 74.55%
# Epoch [32/200], Validation Loss: 0.7135, Validation Acc: 75.32%
# Epoch [33/200], Step [1/391], Loss: 0.6026, Acc: 78.12%
# Epoch [33/200], Step [101/391], Loss: 0.6345, Acc: 75.44%
# Epoch [33/200], Step [201/391], Loss: 0.8351, Acc: 75.23%
# Epoch [33/200], Step [301/391], Loss: 0.6754, Acc: 75.04%
# Epoch [33/200], Validation Loss: 0.7007, Validation Acc: 75.46%
# Epoch [34/200], Step [1/391], Loss: 0.6296, Acc: 76.56%
# Epoch [34/200], Step [101/391], Loss: 0.6881, Acc: 75.73%
# Epoch [34/200], Step [201/391], Loss: 0.6439, Acc: 75.47%
# Epoch [34/200], Step [301/391], Loss: 0.7022, Acc: 75.73%
# Epoch [34/200], Validation Loss: 0.7140, Validation Acc: 75.18%
# Epoch [35/200], Step [1/391], Loss: 0.6931, Acc: 74.22%
# Epoch [35/200], Step [101/391], Loss: 0.7469, Acc: 76.13%
# Epoch [35/200], Step [201/391], Loss: 0.7325, Acc: 75.90%
# Epoch [35/200], Step [301/391], Loss: 0.6172, Acc: 75.85%
# Epoch [35/200], Validation Loss: 0.6935, Validation Acc: 75.69%
# Epoch [36/200], Step [1/391], Loss: 0.7125, Acc: 76.56%
# Epoch [36/200], Step [101/391], Loss: 0.6898, Acc: 76.21%
# Epoch [36/200], Step [201/391], Loss: 0.6102, Acc: 76.40%
# Epoch [36/200], Step [301/391], Loss: 0.6135, Acc: 76.23%
# Epoch [36/200], Validation Loss: 0.7100, Validation Acc: 75.37%
# Epoch [37/200], Step [1/391], Loss: 0.7538, Acc: 77.34%