import torch
import torch.onnx
import torchvision.models as models
from transformers import AutoModelForImageClassification
from torch import nn

# Load the PyTorch model
model_name = "jialicheng/cifar100-resnet-50"
model = AutoModelForImageClassification.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode

# Create a dummy input tensor
dummy_input = torch.randn(1, 3, 224, 224)  # Adjust the size based on your model's input

# Export the model to ONNX format
onnx_file_path = 'model.onnx'  # Replace with your desired ONNX file path
torch.onnx.export(model, dummy_input, onnx_file_path, 
                  export_params=True, 
                  opset_version=11,  # Use the appropriate ONNX opset version
                  do_constant_folding=True,  # Optimize the model
                  input_names=['input'],  # Name of the input layer
                  output_names=['output'],  # Name of the output layer
                  dynamic_axes={'input': {0: 'batch_size'},  # Variable batch size
                                'output': {0: 'batch_size'}})

print(f'Model has been converted to ONNX and saved at {onnx_file_path}') 