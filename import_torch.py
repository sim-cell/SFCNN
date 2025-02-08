import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() ==1:
    print(f"Using {torch.cuda.device_count()} GPUs")