import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForImageClassification
from torch.utils.data import DataLoader

def iterative_pruning_experiment(model, data, config):
    train_loader, val_loader, test_loader = data

    for truncation in range(10):
        prune_pct = truncation / 10
        config["experiment_name"] = config["experiment_name"][:-2] + f"-{truncation}"
        config["l1_weight"] = 0

        # Apply pruning based on the specified type
        if config["prune_type"] == "unstructured":
            mask = unstructured_magnitude_prune(model, prune_pct)
        else:
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
        print("*" * 35)
        print(f"Expected {truncation * 10}% Pruning:")
        print(f"True pruning rate: {zeroed / total:.2%}")
        print(f"Loss, Accuracy: {evaluate(model, test_loader, config)}")
        print("*" * 35)

def unstructured_magnitude_prune(model, prune_pct):
    # Implement unstructured pruning logic here
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=prune_pct)
    return None  # Return mask if needed

def column_magnitude_pruning(model, prune_pct):
    # Implement column magnitude pruning logic here
    return None  # Return mask if needed

def fine_tune(model, train_loader, val_loader, config, mask):
    # Implement fine-tuning logic here
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config["num_epochs"]):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

def evaluate(model, test_loader, config):
    # Implement evaluation logic here
    model.eval()
    total_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  

            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(test_loader.dataset)
    return total_loss / len(test_loader), accuracy  # Return loss and accuracy

# Example usage
if __name__ == "__main__":
    # Load your model and data here
    model_name = "jialicheng/cifar100-resnet-50"
    model = AutoModelForImageClassification.from_pretrained(model_name)

    # Load your data (train_loader, val_loader, test_loader)
    data = (train_loader, val_loader, test_loader)

    # Configuration dictionary
    config = {
        "experiment_name": "pruning_experiment",
        "prune_type": "unstructured",  # or "structured"
        "num_epochs": 5,  # Example number of epochs for fine-tuning
        "l1_weight": 0.01,  # Example weight for L1 regularization
    }

    # Run the iterative pruning experiment
    iterative_pruning_experiment(model, data, config)