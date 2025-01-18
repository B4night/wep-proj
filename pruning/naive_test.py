import os
import torch
import torch.nn as nn

# Example model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1000, 1000)
        self.fc.weight.data = torch.randn(1000, 1000)

    def forward(self, x):
        return self.fc(x)

# Create a model instance
model = SimpleModel()
temp_path = "models/temp_model.pth"
torch.save(model.state_dict(), temp_path)

# Load the model state with weights_only=True
model.load_state_dict(torch.load(temp_path, weights_only=True))

# Zero out 50% of the weights of the model
with torch.no_grad():
    for param in model.parameters():
        if param.dim() > 1:  # Only apply to weight matrices
            mask = torch.rand(param.shape) < 0.5  # Mask to zero out 50% of the weights
            param[mask] = 0

# Save the original dense model
dense_model_path = "dense_model.pth"
torch.save(model.state_dict(), dense_model_path)

# Custom compression function
def compress_model(state_dict):
    compressed_state = {}
    for key, param in state_dict.items():
        if param.dim() > 1:  # Only compress weight matrices
            values = param[param != 0]  # Non-zero values
            indices = torch.nonzero(param, as_tuple=True)  # Indices of non-zero values
            compressed_state[key] = {"values": values, "indices": indices}
        else:
            compressed_state[key] = param  # Store dense biases or 1D tensors as-is
    return compressed_state

# Compress the model
compressed_state = compress_model(model.state_dict())

# Save the compressed model
compressed_model_path = "compressed_model.pth"
torch.save(compressed_state, compressed_model_path)

# Compare file sizes
dense_size = os.path.getsize(dense_model_path) / (1024 ** 2)  # MB
compressed_size = os.path.getsize(compressed_model_path) / (1024 ** 2)  # MB

print(f"Dense Model File Size: {dense_size:.2f} MB")
print(f"Compressed Model File Size: {compressed_size:.2f} MB")
print(f"Compression Ratio: {compressed_size / dense_size:.2f}")

# Reconstruct the model from the compressed state
def decompress_model(compressed_state):
    decompressed_state = {}
    for key, value in compressed_state.items():
        if isinstance(value, dict):  # Compressed weights
            shape = (1000, 1000)  # Known shape of the layer
            tensor = torch.zeros(*shape)
            tensor[value["indices"]] = value["values"]  # Rebuild dense tensor
            decompressed_state[key] = tensor
        else:
            decompressed_state[key] = value  # Biases or 1D tensors
    return decompressed_state

# Load and decompress the model
compressed_state = torch.load(compressed_model_path)
decompressed_state = decompress_model(compressed_state)

# Load into the model
model.load_state_dict(decompressed_state)

# Verify the model works
input_tensor = torch.randn(1, 1000)
output = model(input_tensor)
print("Output shape:", output.shape)