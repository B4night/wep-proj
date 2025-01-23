import torch
import os
from torchvision import models

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

pruned_model_path = "./resnet56_cifar100_0.20_0.00_0.00.pth"
model = torch.load(pruned_model_path, map_location=device)

# 若需要加载 state_dict，请取消下面这一行的注释，并确保适当处理 key 命名等问题
# model.load_state_dict(state_dict, strict=False)

# 打印模型的 dtype 和大小 (从第一个参数获取数据类型)
first_param_dtype = next(model.parameters()).dtype
print(f'Original model dtype: {first_param_dtype}')
print(f'Original model size: {get_model_size(model=model):.2f} MB')

# 转换为 FP16 并移动到相应设备
model = model.half().to(device)

# 打印 FP16 模型的信息 (从第一个参数获取数据类型)
first_param_dtype_fp16 = next(model.parameters()).dtype
print(f'FP16 model dtype: {first_param_dtype_fp16}')
print(f'FP16 model size: {get_model_size(model=model):.2f} MB')

# 如果需要保存模型，请取消下面一行的注释
# torch.save(model.state_dict(), 'cifar100_resnet18_quantized.pth')
