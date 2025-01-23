import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification
# from conf import settings

# Load CIFAR-100 test dataset
D_CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
D_CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(D_CIFAR100_TRAIN_MEAN, D_CIFAR100_TRAIN_STD)
])

test_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# Load the model

# model_name = "jialicheng/cifar100-resnet-50"
# model = AutoModelForImageClassification.from_pretrained(model_name)
# torch.save(model.state_dict(), "models/resne50_cifar100_model.pth")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pruned_model_path = "./resnet56_cifar100_0.20_0.00_0.00.pth"
model = torch.load(pruned_model_path).to(device)


# Move the model to the appropriate device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)  # Move model to GPU if available
model.eval()  # Set the model to evaluation mode

model = model.half()

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
            # if isinstance(outputs, tuple):
            #     outputs = outputs[0]
            # _, preds = outputs
            end_time = time.time()

            inference_times.append(end_time - start_time)

            # loss = criterion(outputs, labels)
            # running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader)
    avg_inference_time = sum(inference_times) / len(inference_times)

    print(f'Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}')

    return accuracy, avg_loss, avg_inference_time

def get_model_size(model):
    temp_path = "models/resnet18_cifar100_model.pth"
    torch.save(model.state_dict(), temp_path)
    size_in_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)  # Clean up the temporary file
    return size_in_mb

def save_model(model, save_path):
    """Save the model's state dictionary to the specified path."""
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

# Example baseline values (replace with your actual baseline results)
baseline_accuracy = 75.00
baseline_inference_time = 0.08
baseline_model_size = 100.0

model_size = get_model_size(model)
print(f'Model Size: {model_size:.2f} MB')

max_accuracy = 0
min_avg_loss = 100
min_avg_inference_time = 100
rounds = 100
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