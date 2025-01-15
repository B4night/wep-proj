# compare.py
import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 如果你不再使用HuggingFace接口，就不需要再 from transformers import ...
# from transformers import AutoModelForImageClassification

# 1. 加载 CIFAR100
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
test_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# 2. 加载两个模型 (原模型 + 量化模型)
model_fp32 = torch.load("cifar100_resnet50_model.pth", map_location='cpu')
model_int8 = torch.load("model_quantized.pth", map_location='cpu')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_fp32.to(device)
# model_int8.to(device)

model_fp32.eval()
model_int8.eval()

criterion = nn.CrossEntropyLoss()

def test_model_performance(model, loader):
    total = 0
    correct = 0
    running_loss = 0.0
    inference_times = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            # images, labels = images.to(device), labels.to(device)

            start_time = time.time()
            outputs = model(images).logits   # HF模型通常是一个包含logits的对象
            end_time = time.time()

            inference_times.append(end_time - start_time)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(loader)
    avg_inference_time = sum(inference_times) / len(inference_times)
    return accuracy, avg_loss, avg_inference_time

def get_model_size(model):
    temp_path = "temp.pth"
    torch.save(model.state_dict(), temp_path)
    size_in_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return size_in_mb

# 3. 测试原模型
print("Testing the original FP32 model...")
acc_fp32, loss_fp32, time_fp32 = test_model_performance(model_fp32, test_loader)
size_fp32 = get_model_size(model_fp32)
print(f"[FP32] Accuracy: {acc_fp32:.2f}%, Loss: {loss_fp32:.4f}, Inference Time: {time_fp32:.4f}s, Size: {size_fp32:.2f}MB")

# 4. 测试量化模型
print("\nTesting the dynamic-quantized model...")
acc_int8, loss_int8, time_int8 = test_model_performance(model_int8, test_loader)
size_int8 = get_model_size(model_int8)
print(f"[INT8] Accuracy: {acc_int8:.2f}%, Loss: {loss_int8:.4f}, Inference Time: {time_int8:.4f}s, Size: {size_int8:.2f}MB")

# 5. 对比
acc_drop = acc_fp32 - acc_int8
speed_improvement = (time_fp32 - time_int8) / time_fp32 * 100 if time_fp32 > 0 else 0
size_reduction = (size_fp32 - size_int8) / size_fp32 * 100 if size_fp32 > 0 else 0

print("\nComparison:")
print(f"Accuracy Drop:      {acc_drop:.2f}%")
print(f"Speed Improvement:  {speed_improvement:.2f}%")
print(f"Size Reduction:     {size_reduction:.2f}%")
