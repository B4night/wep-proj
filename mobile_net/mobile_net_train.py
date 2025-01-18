import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.models import resnet50

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 20
batch_size = 64
learning_rate = 0.001
weight_decay = 1e-4  # Set your desired weight decay value

# CIFAR-100 Data Loading with Data Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Randomly flip images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Load Pre-trained ResNet-50 and Adapt for CIFAR-100
preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.4_224")
model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.4_224")

# Adjust the classifier layer for 100 classes
num_classes = 100
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

model = model.to(device)

# Loss and Optimizer with Weight Decay
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Adaptive Learning Rate Scheduler with Cosine Annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

# Training Loop
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images).logits
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Step the scheduler at the end of each epoch
        scheduler.step()

    print("Finished Training")

# Evaluate Model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Extract logits from the model output
            logits = outputs.logits  # Use .logits to get the logits

            # Get predicted classes
            _, predicted = torch.max(logits, 1)  # Use logits instead of outputs

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Train and Evaluate the Model
train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs)
evaluate_model(model, test_loader)

# Save the Model in Compressed Format
compressed_model_path = "models/mobile_net_cifar100_model.pth"
torch.save(model.state_dict(), compressed_model_path)
print(f"Model saved to {compressed_model_path}")
