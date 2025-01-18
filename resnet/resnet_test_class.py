import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification

# Load CIFAR-100 test dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Randomly flip images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
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

pruned_model_path = "/home/alievak/git/RLPruner-CNN/compressed_model/resnet56_cifar100_0.30_0.00_0.00.pth"
model = torch.load(pruned_model_path).to(device)
# model.load_state_dict(state_dict, strict=False) 
zeroed, total = 0, 0

# Calculate true pruning rate
for name, param in model.named_parameters():
    if param.requires_grad and "weight" in name:
        num_zeros = torch.sum(param == 0).item()
        total_params = param.numel()
        zeroed += num_zeros
        total += total_params

# Output statistics for the k-th iteration
print(f"True pruning rate: {zeroed / total:.2%}")
# def convert_sparse_to_dense(state_dict):
#     for key, value in state_dict.items():
#         if value.is_sparse:
#             state_dict[key] = value.to_dense()
#     return state_dict

# # Convert sparse tensors to dense
# state_dict = convert_sparse_to_dense(state_dict)
# model.load_state_dict(state_dict)

# def convert_weights_to_dense(model):
#     for name, module in model.named_modules():
#         if hasattr(module, 'weight') and module.weight.is_sparse:
#             module.weight = torch.nn.Parameter(module.weight.to_dense())

# # Convert the model's sparse weights to dense format
# convert_weights_to_dense(model)
# Move the model to the appropriate device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)  # Move model to GPU if available
model.eval()  # Set the model to evaluation mode

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
            outputs = model(images).logits
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

print("Testing the compressed model...")
accuracy, avg_loss, avg_inference_time = test_model_performance(model, test_loader)

model_size = get_model_size(model)
print(f'Model Size: {model_size:.2f} MB')

# Example baseline values (replace with your actual baseline results)
baseline_accuracy = 75.00
baseline_inference_time = 0.08
baseline_model_size = 100.0

accuracy_drop = baseline_accuracy - accuracy
speed_improvement = (baseline_inference_time - avg_inference_time) / baseline_inference_time * 100
size_reduction = (baseline_model_size - model_size) / baseline_model_size * 100

print("\nPerformance Comparison:")
print(f'Accuracy Drop: {accuracy_drop:.2f}%')
print(f'Speed Improvement: {speed_improvement:.2f}%')
print(f'Size Reduction: {size_reduction:.2f}%')

# Save the model after testing
save_path = "models/resnet18_cifar100_model.pth" 
save_model(model, save_path)