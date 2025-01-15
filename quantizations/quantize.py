import argparse
import os
import warnings

import torch
import torch.nn as nn
import torch.quantization as quantization
import torch.backends.quantized

from torchvision import transforms, datasets

warnings.filterwarnings("ignore")


def load_entire_model(model_path: str):
    """
    从 .pth 文件加载整个模型（使用 torch.save(model, ...) 保存的）。
    """
    model = torch.load(model_path, map_location='cpu')
    print(f"[INFO] Loaded model from {model_path}, type={type(model)}")
    return model


def fuse_modules_resnet(model: nn.Module):
    """
    针对 torchvision 官方版 ResNet 结构的模块融合。
    若你的模型结构与此不同，需要自行适配。
    """
    # 遍历每个子模块
    for m_name, m in model.named_children():
        if isinstance(m, nn.Sequential):
            # 逐层查找 BasicBlock/Bottleneck 等
            for sub_name, sub_m in m.named_children():
                # 这里 sub_m 可能是 BasicBlock / Bottleneck
                # 通常包含 conv1/bn1/relu, conv2/bn2, optional downsample
                if hasattr(sub_m, 'conv1') and hasattr(sub_m, 'bn1') and hasattr(sub_m, 'relu'):
                    # fuse conv1+bn1+relu
                    quantization.fuse_modules(sub_m, ['conv1', 'bn1', 'relu'], inplace=True)
                if hasattr(sub_m, 'conv2') and hasattr(sub_m, 'bn2'):
                    # fuse conv2+bn2
                    quantization.fuse_modules(sub_m, ['conv2', 'bn2'], inplace=True)
                # 如果是 Bottleneck，可能还有 conv3/bn3 需要 fuse
                # if hasattr(sub_m, 'conv3') and hasattr(sub_m, 'bn3'):
                #     quantization.fuse_modules(sub_m, ['conv3', 'bn3'], inplace=True)
                #
                # 如果有 downsample，可以考虑 fuse，但要看 downsample 里具体包含哪些层
    return model


# def get_calibration_loader(data_root, batch_size=8):
#     """
#     构造一个简单的 DataLoader，用于做量化校准 (forward)。
#     这里示例用 ImageNet 的子集或其他图片目录。
#     你也可以用自己的数据集替代。
#     """
#     # 简单的预处理变换
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#     ])
#     dataset = datasets.ImageFolder(data_root, transform=transform)
#     data_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=0,  # 根据需要修改
#     )
#     return data_loader


def post_training_static_quantization(model: nn.Module,
                                    #   calibration_loader,
                                      device='cpu'):
    """
    后训练静态量化：
      1. 融合(Fuse)
      2. 准备(Prepare)
      3. 校准(Forward)
      4. 转换(Convert)
    """

    # 1. 融合模块
    model.eval()
    model = model.to(device)
    model = fuse_modules_resnet(model)

    # 2. 设置后端和量化配置
    torch.backends.quantized.engine = 'qnnpack'  # 也可用 'fbgemm'
    model.qconfig = quantization.default_qconfig

    # 3. 准备 (插入观察器等)
    print("[INFO] Preparing model for static quantization...")
    model_prepared = quantization.prepare(model, inplace=False)

    # # 4. 校准 (在校准数据上 forward 一遍)
    # print("[INFO] Calibrating...")
    # with torch.no_grad():
    #     for images, _ in calibration_loader:
    #         images = images.to(device)
    #         _ = model_prepared(images)

    # 5. 转换 (把观察器替换为量化算子)
    print("[INFO] Converting model to quantized version...")
    quantized_model = quantization.convert(model_prepared, inplace=False)

    return quantized_model


def save_model(model, output_path):
    """
    保存模型
    """
    torch.save(model, output_path)
    print(f"[INFO] Quantized model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Static Quantization for a CNN model (e.g. ResNet).")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the input .pth model file (saved with torch.save(model, ...)).")
    # parser.add_argument("--calibration-data-root", type=str, required=True,
    #                     help="Root folder of calibration images (ImageFolder style).")
    parser.add_argument("--output-path", type=str, default="resnet_quantized_static.pth",
                        help="Path to output quantized model file.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for calibration data loader.")
    args = parser.parse_args()

    # 1. 加载原始模型
    original_model = load_entire_model(args.model_path)

    # 2. 构造校准 DataLoader
    # calibration_loader = get_calibration_loader(args.calibration_data_root,
                                                # batch_size=args.batch_size)

    # 3. 执行静态量化
    quantized_model = post_training_static_quantization(
        model=original_model,
        # calibration_loader=calibration_loader,
        device='cpu'
    )

    # 4. 保存量化后的模型
    save_model(quantized_model, args.output_path)

    # 5. 对比文件大小
    original_size = os.path.getsize(args.model_path) / 1024 / 1024
    quantized_size = os.path.getsize(args.output_path) / 1024 / 1024
    ratio = original_size / quantized_size if quantized_size else 1

    print(f"[INFO] Original model size:  {original_size:.2f} MB")
    print(f"[INFO] Quantized model size: {quantized_size:.2f} MB")
    print(f"[INFO] Compression ratio:    {ratio:.2f}x")


if __name__ == "__main__":
    main()
