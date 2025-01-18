import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForImageClassification
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
import time
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def iterative_pruning_experiment(model, data, config):
    train_loader, val_loader, test_loader = data

    for truncation in range(0, 5):
        prune_pct = truncation / 10
        config["experiment_name"] = config["experiment_name"][:-2] + f"-{truncation}"
        config["l1_weight"] = 0

        logging.info(f'Starting pruning iteration {truncation + 1} with {prune_pct * 100:.1f}% pruning.')

        # Apply pruning based on the specified type
        mask = column_magnitude_pruning(model, prune_pct)

        # Fine-tune the model after pruning
        fine_tune(model, train_loader, val_loader, config, mask)

        zeroed, total = 0, 0

        # Calculate true pruning rate
        for name, param in model.named_parameters():
            if param.requires_grad and "weight" in name:
                num_zeros = torch.sum(param == 0).item()
                total_params = param.numel()
                zeroed += num_zeros
                total += total_params

        # Output statistics for the k-th iteration
        logging.info(f"Expected {truncation * 10}% Pruning:")
        logging.info(f"True pruning rate: {zeroed / total:.2%}")
        loss, accuracy = evaluate(model, test_loader, config)
        # logging.info(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")

        # Save the model after each pruning iteration
        model_save_path = f"{config['experiment_name']}_pruned_{truncation}.pth"
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"Model saved to {model_save_path}")

        # Show compression metrics
        compression_ratio = (total - zeroed) / total
        logging.info(f"Model compression ratio after iteration {truncation + 1}: {compression_ratio:.2%}")

def column_magnitude_pruning(model, pruning_ratio=0.2):
    """
    Applies column-based structured magnitude pruning on each layer in the model.

    Args:
        model (nn.Module): The model to prune.
        pruning_ratio (float): Fraction of columns to prune per layer.
    """
    mask = {}

    for name, param in model.named_parameters():
        if "weight" in name:  # Modify as needed to include other layer types
            weight = param.data  # Get the weight tensor
            
            if weight.dim() == 2:  # Fully connected layer
                # Compute the L2 norm of each column
                column_norms = torch.norm(weight, p=2, dim=0)  # Shape: [out_features]

                # Determine the number of columns to remove
                num_columns_to_prune = int(pruning_ratio * weight.size(1))

                # Find the indices of the columns with the smallest norms
                _, prune_indices = torch.topk(column_norms, num_columns_to_prune, largest=False)

                # Create a mask to select the remaining columns
                keep_indices = torch.ones(weight.size(1), dtype=torch.bool, device=weight.device)
                keep_indices[prune_indices] = False

                # Reduce the size of the weight matrix by keeping only the selected columns
                pruned_weight = weight[:, keep_indices]

                # Update the weight matrix in place
                param.data = pruned_weight

                # If there are biases, adjust them accordingly (optional)
                bias_name = name.replace("weight", "bias")
                if hasattr(model, bias_name):
                    bias = getattr(model, bias_name)
                    if bias is not None:
                        param_bias = bias.data
                        param_bias = param_bias[keep_indices]
                        setattr(model, bias_name, torch.nn.Parameter(param_bias))


            elif weight.dim() == 4:  # Convolutional layer
                column_norms = torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)  # Shape: [out_channels]

                num_channels_to_prune = int(pruning_ratio * weight.size(0))

                _, prune_indices = torch.topk(column_norms, num_channels_to_prune, largest=False)

                weight[prune_indices, :, :, :] = 0  # Prune channels

            elif weight.dim() == 1:  # 1D tensor case (e.g., biases)
                continue  # Skip pruning for 1D tensors

            else:
                continue  # Skip if the shape is unexpected

            mask[name] = weight != 0  # Create a mask for the pruned weights

    return mask

def apply_mask(model, mask):

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.data *= mask[name]
                param.grad *= mask[name]

def fine_tune(model, train_loader, val_loader, config, mask):
    # Implement fine-tuning logic here
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config["num_epochs"]):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to device
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            if mask:
                apply_mask(model, mask)
            optimizer.step()

        logging.info(f'Epoch [{epoch + 1}/{config["num_epochs"]}] completed.')

def evaluate(model, test_loader, config):
    # Implement evaluation logic here
    model.eval()
    total = 0
    total_loss = 0
    correct = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    inference_times = []
    temp_path = "models/temp_model.pth"
    torch.save(model.state_dict(), temp_path)
    model.load_state_dict(torch.load(temp_path))
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  

            start_time = time.time()
            outputs = model(images).logits
            end_time = time.time()

            inference_times.append(end_time - start_time)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / len(test_loader.dataset)
    avg_loss = running_loss / len(test_loader)
    avg_inference_time = sum(inference_times) / len(inference_times)
    baseline_accuracy = 75.00
    baseline_inference_time = 0.08
    # baseline_model_size = 100.0

    accuracy_drop = baseline_accuracy - accuracy
    speed_improvement = (baseline_inference_time - avg_inference_time) / baseline_inference_time * 100
    # size_reduction = (baseline_model_size - model_size) / baseline_model_size * 100
    print(f'Accuracy: {accuracy:.2f}')
    print("\nPerformance Comparison:")
    print(f'Accuracy Drop: {accuracy_drop:.2f}%')
    print(f'Speed Improvement: {speed_improvement:.2f}%')
    # print(f'Size Reduction: {size_reduction:.2f}%')

    print(f'Accuracy: {accuracy:.2f}%, Average Loss: {avg_loss:.4f}')
    return total_loss / len(test_loader), accuracy  # Return loss and accuracy

# Example usage
if __name__ == "__main__":
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load your model and data here
    model_name = "jialicheng/cifar100-resnet-50"
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)  # Move model to device

    # Define your batch size
    batch_size = 64  # Adjust this based on your requirements

    # Define the data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Randomly flip images
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))   
    ])

    # Load the CIFAR100 dataset
    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Assuming val_loader is defined similarly to train_loader
    val_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)  # Example for validation

    # Load your data (train_loader, val_loader, test_loader)
    data = (train_loader, val_loader, test_loader)

    # Configuration dictionary
    config = {
        "experiment_name": "pruning_experiment",
        "prune_type": "structured",  # or "structured"
        "num_epochs": 1,  # Example number of epochs for fine-tuning
        "l1_weight": 0.01,  # Example weight for L1 regularization
    }

    # Run the iterative pruning experiment
    iterative_pruning_experiment(model, data, config)