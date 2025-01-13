import torch
import timm
from torch import nn

save_path = "compressed_model.pth"

model = timm.create_model("hf_hub:edadaltocg/resnet50_cifar100", pretrained=True)
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

torch.save(model.state_dict(), save_path)  # Save only the state_dict

print(f"Pretrained ResNet-50 model saved to {save_path}")
