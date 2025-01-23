import torch
import os
import timm

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

# 创建模型（与训练时一致）
model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=100)
model = model.to(device)

# 加载 state_dict 并处理多卡训练（DataParallel）可能导致的 'module.' 前缀
state_dict = torch.load('cifar100_resnet18.pth', map_location=device)
# 去掉多卡训练的 'module.' 前缀（如果存在）
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# 加载模型参数，忽略多余或缺失的层
model.load_state_dict(state_dict, strict=False)

# 打印模型中第一个权重的 dtype，模型本身没有 dtype 属性
first_param_dtype = next(model.parameters()).dtype
print(f'Original model dtype: {first_param_dtype}')
print(f'Original model size: {get_model_size(model=model):.2f} MB')

# 将模型转换为 FP16 并移动到设备
model = model.half().to(device)

# 打印 FP16 模型的信息
first_param_dtype_fp16 = next(model.parameters()).dtype
print(f'FP16 model dtype: {first_param_dtype_fp16}')
print(f'FP16 model size: {get_model_size(model=model):.2f} MB')

torch.save(model.state_dict(), 'cifar100_resnet18_quantized.pth')