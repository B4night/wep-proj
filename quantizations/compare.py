import os
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")

torch.random.manual_seed(42)

def test_model_performance(model, loader, device="cpu"):
    """
    Measure accuracy, average loss, and average inference time.
    This function runs inference on the provided data loader and
    returns accuracy, average loss, and average time per batch.
    """
    total = 0
    correct = 0
    running_loss = 0.0
    inference_times = []

    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            start_time = time.time()
            
            # If you're using HuggingFace's ResNetForImageClassification:
            # "outputs" is a ModelOutput object, containing "logits"
            if model.dtype == torch.float16:
                images = images.half()
                
            outputs = model(images).logits
            
            end_time = time.time()
            inference_times.append(end_time - start_time)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate the number of correct predictions
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # break

    # Compute metrics
    print(f'Length of inference times: {len(inference_times)}')
    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(loader)
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    return accuracy, avg_loss, avg_inference_time

def get_model_size(model, temp_path="temp.pth"):
    """
    Roughly estimate the model's size on disk in MB using state_dict().
    """
    torch.save(model.state_dict(), temp_path)
    size_in_mb = os.path.getsize(temp_path) / (1024 * 1024)
    os.remove(temp_path)
    return size_in_mb

def main():
    # 1. Prepare CIFAR-100 test set
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载模型
    model_fp32_path = "./model.pth"
    model_qnt_path = "./model_qnt.pth"
    
    if not os.path.exists(model_fp32_path):
        raise FileNotFoundError(f"[ERROR] File not found: {model_fp32_path}")
    if not os.path.exists(model_qnt_path):
        raise FileNotFoundError(f"[ERROR] File not found: {model_qnt_path}")

    # 加载模型
    print("Loading models...")
    model_fp32 = torch.load(model_fp32_path, map_location=device)
    model_fp32.eval()
    
    # 加载量化模型
    try:
        model_qnt = torch.load(model_qnt_path, map_location=device)
        model_qnt.eval()
        # 调试信息
        # print("Model type:", type(model_qnt))
        # print("Model structure:", model_qnt)
        # for name, module in model_qnt.named_modules():
        #     print(f"Layer: {name}, Type: {type(module)}")
    except Exception as e:
        print(f"Error loading quantized model: {e}")
        raise

    print("Models loaded successfully")

    # Print how many instances in the loader
    print(f"Number of instances in the loader: {len(test_loader)}")

    # 3. Test FP32 model
    print("\nTesting the original FP32 model...")
    try:
        acc_fp32, loss_fp32, time_fp32 = test_model_performance(model_fp32, test_loader, device=device)
        size_fp32 = get_model_size(model_fp32)
        print(f"[FP32] Accuracy:      {acc_fp32:.2f}%")
        print(f"[FP32] Avg. Loss:     {loss_fp32:.4f}")
        print(f"[FP32] Inference Time per Batch: {time_fp32:.4f}s")
        print(f"[FP32] Model Size:    {size_fp32:.2f} MB")
    except Exception as e:
        print(f"Error testing FP32 model: {e}")
        raise

    # 4. Test quantized model
    print("\nTesting the quantized model...")
    try:
        acc_int8, loss_int8, time_int8 = test_model_performance(model_qnt, test_loader, device=device)
        size_int8 = get_model_size(model_qnt)
        print(f"[INT8] Accuracy:      {acc_int8:.2f}%")
        print(f"[INT8] Avg. Loss:     {loss_int8:.4f}")
        print(f"[INT8] Inference Time per Batch: {time_int8:.4f}s")
        print(f"[INT8] Model Size:    {size_int8:.2f} MB")
    except Exception as e:
        print(f"Error testing quantized model: {e}")
        raise

    # 5. Compare results
    acc_drop = acc_fp32 - acc_int8
    speed_improvement = ((time_fp32 - time_int8) / time_fp32 * 100) if time_fp32 > 0 else 0
    size_reduction = ((size_fp32 - size_int8) / size_fp32 * 100) if size_fp32 > 0 else 0

    print("\nComparison:")
    print(f" - Accuracy Drop:      {acc_drop:.2f}%")
    print(f" - Speed Improvement:  {speed_improvement:.2f}%  (lower inference time is better)")
    print(f" - Size Reduction:     {size_reduction:.2f}%  (smaller size is better)")

if __name__ == "__main__":
    main()
