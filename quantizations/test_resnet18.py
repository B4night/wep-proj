import torch
import os
from torchvision import models
from torch import nn

def get_model_size(model):
    """计算模型的文件大小（MB）"""
    temp_path = "temp.pth"
    torch.save(model.state_dict(), temp_path)
    size_in_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)  # 删除临时文件
    return size_in_mb

# 自动选择设备（CPU 或 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 创建 ResNet18 模型，并修改全连接层（100分类）
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

# 加载模型权重（请替换为正确的权重路径）
weight_path = 'cifar100_resnet18.pth'  # 替换为实际路径
if os.path.exists(weight_path):
    # 加载权重，处理多卡训练的 'module.' 前缀
    state_dict = torch.load(weight_path, map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # 加载权重，允许部分加载（避免因维度不一致导致报错）
    model.load_state_dict(state_dict, strict=False)
    print("Model weights loaded successfully!")
else:
    print(f"Weight file not found at: {weight_path}")

# 打印模型的 dtype 和大小
first_param_dtype = next(model.parameters()).dtype
print(f'Original model dtype: {first_param_dtype}')
print(f'Original model size: {get_model_size(model=model):.2f} MB')

# 转换为 FP16 并移动到相应设备
model = model.half().to(device)

# 打印 FP16 模型的信息
first_param_dtype_fp16 = next(model.parameters()).dtype
print(f'FP16 model dtype: {first_param_dtype_fp16}')
print(f'FP16 model size: {get_model_size(model=model):.2f} MB')

torch.save(model.state_dict(), 'cifar100_resnet18_quantized.pth')
