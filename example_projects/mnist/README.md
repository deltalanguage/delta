# MNIST Example - Delta Language

A simple feedforward neural network for MNIST digit classification, 
demonstrating the Delta language's differentiable programming features.

## Model Architecture

```
Input (784) → Linear+ReLU (256) → Linear+ReLU (128) → Linear (10) → Softmax
```

**Expected accuracy:** ~98% after 10 epochs

## Files

- `model.delta` - The Delta language model definition
- `train.py` - Training script (supports both Delta and PyTorch backends)
- `README.md` - This file

## Quick Start

### Using PyTorch Backend (immediate)

```bash
cd example_projects/mnist
python train.py --backend pytorch
```

### Using Delta Compiler

```bash
cd example_projects/mnist
python train.py --backend delta
```

## Delta Language Features Demonstrated

### Parameter Declarations
```delta
param W1: Tensor[Float, 784, 256] = randn(784, 256) * 0.01;
```

### Observed Data
```delta
obs images: Tensor[Float, batch, 784];
obs labels: Tensor[Int, batch];
```

### Functions
```delta
fn forward(x: Tensor) -> Tensor {
    let h1 = relu(matmul(x, W1) + b1);
    // ...
}
```

### Training Blocks with Optimizer
```delta
train with Adam(lr=0.001) for 10 epochs {
    let logits = forward(images);
    let loss = cross_entropy(logits, labels);
}
```

### Constraints (Differentiable)
```delta
constraint sum(W1 ** 2) + sum(W2 ** 2) < 100.0 weight 0.0001;
```

### Inference Blocks
```delta
infer {
    let logits = forward(images);
    return argmax(softmax(logits), dim=1);
}
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision (for MNIST dataset)

Install dependencies:
```bash
pip install torch torchvision
```

## Sample Output

```
================================================================
MNIST Training with PyTorch
(Delta model architecture implemented in PyTorch)
================================================================

Device: cuda

Loading MNIST dataset...
Training samples: 60000
Test samples: 10000

Model architecture:
MNISTNet(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=10, bias=True)
)

Total parameters: 235,146

Training for 10 epochs...
------------------------------------------------------------

Epoch 1/10
  Train Loss: 0.3521, Train Acc: 89.45%
  Test Loss:  0.1823, Test Acc:  94.62%
  ✓ New best model saved!

...

Epoch 10/10
  Train Loss: 0.0412, Train Acc: 98.91%
  Test Loss:  0.0789, Test Acc:  97.85%

================================================================
Training complete!
Best test accuracy: 98.12%
================================================================
```

## Next Steps

1. Try different architectures in `model.delta`
2. Add dropout for regularization
3. Use convolutional layers for better accuracy
4. Experiment with different constraints
