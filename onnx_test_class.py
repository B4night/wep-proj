import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import time
import onnxruntime as ort
import os
from torch.utils.data import DataLoader
import numpy as np

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

# Load ONNX model with GPU support
onnx_model_path = "model.onnx"  # Replace with your ONNX model path
ort_session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

# Define the criterion (CrossEntropyLoss)
criterion = nn.CrossEntropyLoss()

# Function to measure inference speed, memory usage, and accuracy
def test_model_performance(ort_session, test_loader):
    total = 0
    correct = 0
    running_loss = 0.0
    inference_times = []

    with torch.no_grad():
        for images, labels in test_loader:
            # Move images and labels to GPU
            images = images.cuda()  # Move images to GPU
            labels = labels.cuda()  # Move labels to GPU

            # Convert images to NumPy for ONNX Runtime
            # images_np = images.cpu().numpy()  # Convert to NumPy
            # Measure inference time
            ort_inputs = {ort_session.get_inputs()[0].name: images}

            start_time = time.perf_counter()
            outputs = ort_session.run(None, ort_inputs)
            end_time = time.perf_counter()
            
            inference_times.append(end_time - start_time)

            # Convert outputs to tensor for loss calculation
            outputs_tensor = torch.tensor(outputs[0])  # Keep outputs on CPU

            # Calculate loss (if using CPU for loss calculation)
            loss = criterion(outputs_tensor, labels.cpu())  # Move labels to CPU if needed
            running_loss += loss.item()

            # Get predicted classes
            _, predicted = torch.max(outputs_tensor, 1)
            total += labels.size(0)  # Use size(0) for batch size
            correct += (predicted == labels).sum().item()  # Use .item() to get the value

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader)
    avg_inference_time = sum(inference_times) / len(inference_times)

    print(f'Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}')
    print(f'Average Inference Time per Batch: {avg_inference_time:.4f} seconds')

    return accuracy, avg_loss, avg_inference_time

# Function to measure model size in MB
def get_model_size(onnx_model_path):
    size_in_mb = os.path.getsize(onnx_model_path) / (1024 * 1024)
    return size_in_mb

# Test the model and get performance metrics
print("Testing the ONNX model...")
accuracy, avg_loss, avg_inference_time = test_model_performance(ort_session, test_loader)

# Get model size
model_size = get_model_size(onnx_model_path)
print(f'Model Size: {model_size:.2f} MB')

# Example baseline values (replace with your actual baseline results)
baseline_accuracy = 75.00  # Example baseline accuracy in %
baseline_inference_time = 0.08  # Example baseline inference time in seconds
baseline_model_size = 100.0  # Example baseline model size in MB

# Calculate performance improvements
accuracy_drop = baseline_accuracy - accuracy
speed_improvement = (baseline_inference_time - avg_inference_time) / baseline_inference_time * 100
size_reduction = (baseline_model_size - model_size) / baseline_model_size * 100

print("\nPerformance Comparison:")
print(f'Accuracy Drop: {accuracy_drop:.2f}%')
print(f'Speed Improvement: {speed_improvement:.2f}%')
print(f'Size Reduction: {size_reduction:.2f}%') 
