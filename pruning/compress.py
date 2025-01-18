import torch
import gzip
import shutil
import os
from transformers import AutoModelForImageClassification


# Save the model
# torch.save(model.state_dict(), "model.pth")
model_name = "jialicheng/cifar100-resnet-50"
model = AutoModelForImageClassification.from_pretrained(model_name)
# Compress the file using gzip
with open("pruning_experime-3_pruned_3.pth", "rb") as f_in:
    with gzip.open("compressed_model.pth.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

with open("cifar100_resnet50_model.pth", "rb") as f_in:
    with gzip.open("original_model.pth.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

# Decompress the file for loading
with gzip.open("original_model.pth.gz", "rb") as f_in:
    with open("model_decompressed.pth", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

# Load the decompressed model
state_dict = torch.load("model_decompressed.pth")
model.load_state_dict(state_dict)

def get_file_size_in_mb(file_path):
    return os.path.getsize(file_path) / (1024 ** 2)

original_size = get_file_size_in_mb("original_model.pth.gz")
compressed_size = get_file_size_in_mb("compressed_model.pth.gz")
original_size_pth = get_file_size_in_mb("cifar100_resnet50_model.pth")

print(f"Original file size: {original_size:.2f} MB")
print(f"Compressed file size: {compressed_size:.2f} MB")
print(f"Original file size: {original_size_pth:.2f} MB")