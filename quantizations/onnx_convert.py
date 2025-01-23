import torch
import torchvision.models as models
import onnx
import onnxruntime as ort
import numpy as np
import torch.nn as nn
import os
import timm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pruned_model_path = "./resnet56_cifar100_0.20_0.00_0.00.pth"
model = torch.load(pruned_model_path).to(device)

model.to(device)  # Move model to GPU if available
model.half()
model.eval()  # Set the model to evaluation mode

# Step 3: Create a Dummy Input in FP16
# mobilenetv3_small expects an input of shape [batch_size, 3, 32, 32]
dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float32).half().to(device)

# TODO: Step 4: Export the Model to ONNX in FP16
onnx_path = "resnet56.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"mobilenetv3_small model has been converted to ONNX (FP16) and saved at {onnx_path}")

# Step 5: Verify the ONNX Model
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Step 6: Run Inference with ONNX Runtime in FP16
# Initialize ONNX Runtime session with FP16 precision
ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])

# Convert the dummy input to NumPy (FP16)
dummy_input_np = dummy_input.cpu().numpy()  # Move to CPU and convert to NumPy array

# Run inference using ONNX Runtime
ort_inputs = {'input': dummy_input_np}  # Input dictionary for ONNX runtime
ort_outs = ort_session.run(None, ort_inputs)

print("Inference with ONNX Runtime (FP16) completed successfully!")

# Step 7: Compare ONNX Output with PyTorch Output
# Run inference on the PyTorch model
with torch.no_grad():
    torch_out = model(dummy_input)  # Keep the input on the same device as the model

# Convert PyTorch output to NumPy for comparison
torch_out_np = torch_out.cpu().numpy()  # Move to CPU and convert to NumPy array

# Compare outputs
if np.allclose(torch_out_np, ort_outs[0], atol=1e-2):  # Adjust tolerance for FP16
    print("ONNX FP16 output matches PyTorch FP16 output!")
else:
    print("ONNX FP16 output does not match PyTorch FP16 output. Check for potential issues.")