import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification
from torchvision import models
import timm

# Load CIFAR-100 test dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

test_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Load the model
# model_name = "jialicheng/cifar100-resnet-50"
# model = AutoModelForImageClassification.from_pretrained(model_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=100)
model = model.to(device)

# 加载 state_dict 并处理多卡训练（DataParallel）可能导致的 'module.' 前缀
state_dict = torch.load('cifar100_mobilenetv3_large_100_model.pth', map_location=device)
# 去掉多卡训练的 'module.' 前缀（如果存在）
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# 加载模型参数，忽略多余或缺失的层
model.load_state_dict(state_dict, strict=False)

model = model.half()

# Move the model to the appropriate device
model.eval()  # Set the model to evaluation mode

first_param_dtype = next(model.parameters()).dtype
print(f'Original model dtype: {first_param_dtype}')

criterion = nn.CrossEntropyLoss()

def test_model_performance(model, test_loader):
    total = 0
    correct = 0
    running_loss = 0.0
    inference_times = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  

            start_time = time.time()
            images = images.half()
            outputs = model(images)
            end_time = time.time()

            inference_times.append(end_time - start_time)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader)
    avg_inference_time = sum(inference_times) / len(inference_times)

    print(f'Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}')

    return accuracy, avg_loss, avg_inference_time

def get_model_size(model):
    temp_path = "temp.pth"
    torch.save(model.state_dict(), temp_path)
    size_in_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)  # Clean up the temporary file
    return size_in_mb

def save_model(model, save_path):
    """Save the model's state dictionary to the specified path."""
    torch.save(model, save_path)
    # torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

baseline_accuracy = 71.00
baseline_inference_time = 0.05
baseline_model_size = 20.0


model_size = get_model_size(model)
print(f'Model Size: {model_size:.2f} MB')

max_accuracy = 0
min_avg_loss = 100
min_avg_inference_time = 100
rounds = 20
for i in range(rounds):
    accuracy, avg_loss, avg_inference_time = test_model_performance(model, test_loader)
    max_accuracy = max(accuracy, max_accuracy)
    min_avg_loss = min(min_avg_loss, avg_loss)
    min_avg_inference_time = min(min_avg_inference_time, avg_inference_time)

speed_improvement = (baseline_inference_time - min_avg_inference_time) / baseline_inference_time * 100
size_reduction = (baseline_model_size - model_size) / baseline_model_size * 100

print("\nPerformance Comparison:")
print(f'Accuracy: {max_accuracy:.2f}%')
print(f'Speed Improvement: {speed_improvement:.2f}%')
print(f'Size Reduction: {size_reduction:.2f}%')
print(f'weighted score: {0.4 * speed_improvement + 0.3 * size_reduction + 0.3 * max_accuracy}')