# RL Pruned Model Performance Testing 

Evaluation of pruned model's accuracy, inference time, and size, and compares the results to baseline values.

- **Model Testing**: Tests all `.pth` models in the `./compressed_model/` folder.
- **Performance Metrics**: Measures accuracy, average loss, inference time, and model size.
- **Comparison to Baseline**: Compares each model's performance to baseline values for accuracy, inference time, and model size.

## Evaluation
```bash
conda create -n RL_Pruner python=3.10 -y
conda activate RL_Pruner
pip install -r requirements.txt
```


## Evaluation
```bash
python test_prunded_model.py
```
