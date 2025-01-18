import torch
from transformers import AutoModelForImageClassification

# Specify the path to save the model
save_path = "jialicheng_resnet50_cifar100_model.pth"

# Load the pretrained model
model_name = "jialicheng/cifar100-resnet-50"
model = AutoModelForImageClassification.from_pretrained(model_name)

# Optionally, modify the model architecture if needed
# For example, if you want to change the first convolutional layer:
# model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

# Save only the state_dict of the model
torch.save(model, save_path)
print(f"Pretrained ResNet-50 model saved to {save_path}")

# Load the model state_dict
loaded_model = AutoModelForImageClassification.from_pretrained(model_name)  # Initialize a new model instance
loaded_model = torch.load(save_path, weights_only=True)  # Load the saved state_dict with weights_only=True
loaded_model.eval()  # Set the model to evaluation mode

print(f"Model loaded from {save_path} and is ready for evaluation.")
