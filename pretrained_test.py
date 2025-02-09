import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
#from modules import *
#from modules_no_expansion import * # Use this for the model without expansion
from convnext import *
from vit_pytorch import ViT
from modules import *

DATA_DIR = './data'
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# same steps as training just for testing
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),  # CIFAR-10 stats
])

test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

def evaluate_model(model, test_loader, device):
    model.eval()  # eval mode
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    return accuracy, avg_loss

# loading pretrained models 
def load_model(model_class, model_path, num_classes=10):
    model = model_class(num_classes=num_classes) 
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)  # Use strict=False to allow loading with mismatches
    model.to(DEVICE) 
    return model

models_to_test = [
    #{"name": "SFCNN_T", "class": SFCNN, "path": "models/sfcnn_tiny_cifar10.pth"},
    #{"name": "Baseline_CNN", "class": CNN, "path": "models/Baseline-CNN-Small.pth"},
    #{"name": "ConvNeXT","class": ConvNeXt, "path": "models/convnext_cifar10.pth"},
    {"name": "ViT", "class": ViT, "path": "models/Baseline-ViT.pth"}, 
]


results = {}
for model_info in models_to_test:
    print(f"Evaluating {model_info['name']}...")

    if model_info["name"]=="ConvNeXT":
        model = ConvNeXt(NUM_CLASSES,
                 channel_list=[64, 128, 256, 512],
                 num_blocks_list=[2, 2, 2, 2],
                 kernel_size=7, patch_size=1,
                 res_p_drop=0.).to(DEVICE)

        model.load_state_dict(torch.load(model_info["path"], map_location=DEVICE))
        accuracy, avg_loss = evaluate_model(model, test_loader, DEVICE)
        results[model_info["name"]] = {"accuracy": accuracy, "avg_loss": avg_loss}
        print(f"{model_info['name']} - Accuracy: {accuracy:.2f}%, Avg Loss: {avg_loss:.4f}")

        continue

    if model_info["name"] == "ViT":
        model = ViT(
            image_size=32,
            patch_size=4,
            num_classes=10,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        ).to(DEVICE)

        model.load_state_dict(torch.load(model_info["path"], map_location=DEVICE))
        accuracy, avg_loss = evaluate_model(model, test_loader, DEVICE)
        results[model_info["name"]] = {"accuracy": accuracy, "avg_loss": avg_loss}
        print(f"{model_info['name']} - Accuracy: {accuracy:.2f}%, Avg Loss: {avg_loss:.4f}")

        continue
    
    model = load_model(model_info["class"], model_info["path"])
    accuracy, avg_loss = evaluate_model(model, test_loader, DEVICE)
    results[model_info["name"]] = {"accuracy": accuracy, "avg_loss": avg_loss}
    print(f"{model_info['name']} - Accuracy: {accuracy:.2f}%, Avg Loss: {avg_loss:.4f}")

print("\nTest Results:")
for model_name, metrics in results.items():
    print(f"{model_name}: Accuracy = {metrics['accuracy']:.2f}%, Avg Loss = {metrics['avg_loss']:.4f}")