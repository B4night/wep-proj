import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification, BitsAndBytesConfig

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
model_name = "jialicheng/cifar100-resnet-50"
model_origin = AutoModelForImageClassification.from_pretrained(model_name, device_map='auto')
model_4bit = AutoModelForImageClassification.from_pretrained(model_name, load_in_4bit=True, device_map='auto')

# Move the model to the appropriate device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
model_origin.to(device)
model_4bit.to(device)

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
            if model.dtype == torch.float16:
                images = images.half()
            outputs = model(images).logits
            end_time = time.time()

            inference_times.append(end_time - start_time)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Length of inference times: {len(inference_times)}')
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
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

def print_info(model):
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


# # Test original model
# print("Original Model:")
# print_info(model_origin)

# Test the 4-bit model
print("\n4-bit Model:")
print_info(model_4bit)

# model_qnt = model_origin.to(dtype=torch.int8)
# print_info(model_qnt)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=3.0
)

model_qnt = AutoModelForImageClassification.from_pretrained(model_name, device_map='auto', quantization_config=bnb_config)
print_info(model_qnt)

# torch.save(model_origin, "model.pth")
# torch.save(model_4bit, "model_4bit.pth")
torch.save(model_qnt, "model_qnt.pth")