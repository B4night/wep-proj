# Model Performance Testing Script

This script is designed to test the performance of pruned PyTorch models on the CIFAR-100 dataset. It evaluates each model's accuracy, inference time, and size, and compares the results to baseline values.

- **Model Testing**: Tests all `.pth` models in the `./compressed_model/` folder.
- **Performance Metrics**: Measures accuracy, average loss, inference time, and model size.
- **Comparison to Baseline**: Compares each model's performance to baseline values for accuracy, inference time, and model size.

## Evaluation
```bash
python test_prunded_model.py
```
