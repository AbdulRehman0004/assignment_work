# Compact CNN + ONNX Runtime + INT8 Quantization (CIFAR-10)

This repository contains my assignment notebook implementation for building and optimizing a compact CNN on CIFAR-10.

## Objective
- Build an FP32 compact CNN with model size `< 500 KB`
- Reach test accuracy `>= 85%`
- Export to ONNX and benchmark CPU inference
- Apply post-training quantization (static INT8) with minimal accuracy drop

## Main Notebook
- `AbdulRehman.ipynb`

## Environment
- Python: `3.12.7` (notebook kernel)
- GPU used for training: None (CPU-only setup)
- CPU used for inference timing: Apple M2

## Method Summary
1. Designed a depthwise-separable CNN (`CompactCIFARNet`) for CIFAR-10.
2. Trained in PyTorch with SGD + momentum, cosine LR schedule, and light regularization.
3. Exported FP32 model to ONNX and evaluated with ONNX Runtime on CPU.
4. Performed static INT8 quantization (QDQ, per-channel weights, MinMax calibration).
5. Included optional dynamic INT8 quantization comparison.

## Results
| Metric | FP32 (PyTorch) | FP32 (ONNX) | INT8 Static | INT8 Dynamic (Optional) |
|---|---:|---:|---:|---:|
| Model Size (KB) | 430.82 | 431.20 | 157.27 | 428.77 |
| Test Accuracy (%) | 89.23 | 89.23 | 89.15 | 89.24 |
| Accuracy Drop vs FP32 ONNX (%) | - | 0.00 | 0.08 | -0.01 |
| Inference Time (ms/batch, CPU, batch=128) | 132.66 | 26.04 | 15.94 | 28.28 |

## Key Takeaways
- The compact CNN satisfies both constraints: size `< 500 KB` and accuracy `> 85%`.
- ONNX Runtime significantly improves CPU inference speed over original FP32 PyTorch timing.
- Static INT8 quantization provides the best trade-off in this assignment: major model-size reduction and faster inference with negligible accuracy loss.

## Repository Files
- `AbdulRehman.ipynb`: complete assignment notebook (all code in-notebook)
- `requirements.txt`: Python dependencies

## Reproducibility
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook AbdulRehman.ipynb
```

Run notebook cells in order to reproduce training, export, quantization, and benchmark outputs.
