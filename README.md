# Delta

[![PyPI version](https://badge.fury.io/py/delta-lang.svg)](https://pypi.org/project/delta-lang/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Differentiable, Constraint-Oriented Programming Language**

[**Website**](https://deltalanguage.org) | [**Documentation**](https://deltalanguage.org/docs) | [**Get Started**](https://deltalanguage.org/docs/quick-start)

Delta is a compiled language that transforms *learning intent* into optimized tensor programs. Instead of manually writing loss functions and training loops, you declare what you want your model to learn through constraints—Delta compiles this into efficient PyTorch code.

```delta
param weights = randn(784, 10);
obs images: Tensor[Float];
obs labels: Tensor[Int];

fn predict(x: Tensor[Float]) -> Tensor[Float] {
    softmax(x @ weights)
}

let pred = predict(images);
require argmax(pred) == labels;
```

This compiles to an optimized PyTorch FX graph with automatic differentiation, constraint-based loss computation, and mode-specific specialization.

---

## Key Features

### Constraint-Based Learning
Define *what* your model should learn, not *how*. Delta compiles constraints into differentiable objectives automatically.

```delta
// Hard constraint: predictions must match labels
require pred == target;

// Soft constraint: prefer smaller weights (regularization)
prefer norm(weights) < 1.0 weight 0.01;
```

### Role Annotations
Explicitly mark learnable parameters vs. observed data. Delta tracks gradient flow and validates differentiability at compile time.

```delta
param theta = randn(10, 5);   // Learnable - gradients flow here
obs x: Tensor[Float];          // Fixed input - no gradients
const scale = 0.1;             // Compile-time constant
```

### Mode Specialization
Write once, get optimized code for training, inference, and analysis:

```delta
train {
    // Only runs during training
    let dropout_mask = rand() > 0.5;
}

infer {
    // Only runs during inference - dropout disabled
    let dropout_mask = ones();
}
```

### Differentiable Control Flow
Conditionals compile to soft, differentiable operations by default:

```delta
// Compiles to: sigmoid((x - 0) / temperature) * a + (1 - sigmoid(...)) * b
if x > 0 temperature 0.1 { a } else { b }
```

---

## Installation

```bash
pip install delta-lang
```

### Requirements
- Python 3.10+
- PyTorch 2.0+ (installed automatically)

### From Source

```bash
git clone https://github.com/deltalanguage/delta.git
cd delta
pip install -e .
```

---

## Quick Start

### 1. Write a Delta Program

Create `model.delta`:

```delta
let model = Linear(10, 5);
obs x: Tensor[Float];
obs y: Tensor[Float];

fn forward(input: Tensor[Float]) -> Tensor[Float] {
    relu(model(input))
}

let pred = forward(x);
require sum((pred - y) ** 2) == 0;
```

### 2. Compile and Train

```python
import delta
import torch

# Compile the Delta program
model = delta.compile("model.delta")

# Prepare data
x_train = torch.randn(100, 10)
y_train = torch.randn(100, 5)

# Train
model.fit(
    {"x": x_train, "y": y_train},
    epochs=100,
    lr=0.01
)

# Inference
model.eval()
x_test = torch.randn(10, 10)  # Test data with 10 samples
predictions = model(x_test)
print(predictions)
```

---

## Examples

### MNIST Classification

```delta
// model.delta
let fc1 = Linear(784, 256);
let fc2 = Linear(256, 128);
let fc3 = Linear(128, 10);

fn forward(x: Tensor) -> Tensor {
    fc3(relu(fc2(relu(fc1(x)))))
}
```

```python
import delta
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Flatten images to 784-dim vectors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

model = delta.compile("model.delta")
model.fit(train_loader, epochs=10, lr=0.001)
```

### Character-Level Transformer

See `example_projects/transformer/` for a complete implementation of a GPT-style language model in Delta.

---

## Language Reference

### Types

| Type | Description |
|------|-------------|
| `Int` | Integer |
| `Float` | Floating-point |
| `Bool` | Boolean |
| `Tensor[T]` | Tensor with element type T |
| `Tensor[Float, N, M]` | Tensor with shape |

### Declarations

```delta
param name = initializer;     // Learnable parameter
obs name: Type;               // Observed input
const name = value;           // Compile-time constant
let name = expr;              // Local binding
```

### Constraints

```delta
require expr == target;                    // Hard equality
require expr > 0;                          // Hard inequality
prefer expr < 1.0 weight 0.5;              // Soft constraint
prefer expr == target weight 1.0 slack s;  // With slack variable
```

### Learn Blocks

```delta
learn {
    require forward(x) == y;
} with epochs = 10, batch_size = 32, optimizer = Adam(lr=1e-3);
```
Learn blocks define training objectives and configurations. The Delta compiler synthesizes a training loop loss function from the constraints.

### Control Flow

```delta
if condition { then_expr } else { else_expr }
if condition temperature 0.1 { ... }  // Soft/differentiable
for i in range(n) { ... }
// Note: while loops and recursion are currently not supported
```

### Mode Blocks

```delta
train { ... }     // Training-only code
infer { ... }     // Inference-only code
analyze { ... }   // Analysis-only code
non_diff { ... }  // Non-differentiable (hard control flow allowed)
```

---

## How It Works

Delta is a **staged compiler** that:

1. **Parses** Delta source into an AST
2. **Infers types** and validates roles/effects
3. **Lowers to SIR** (Semantic IR) — a differentiable-aware representation
4. **Relaxes** hard operations into soft, differentiable alternatives
5. **Compiles constraints** into penalty terms
6. **Specializes** for the execution mode (train/infer)
7. **Lowers to PyTorch FX** for execution

Delta never interprets tensor operations—it compiles everything to optimized PyTorch graphs.

```
Delta Source → AST → Typed AST → SIR → Optimized SIR → PyTorch FX → Execution
```

---

## Architecture

```
delta/
├── frontend/          # Lexer, parser, AST
├── types/             # Type system and inference
├── ir/                # Semantic IR (SIR) definition
├── passes/            # Compiler passes (relaxation, optimization)
├── backend/           # PyTorch FX lowering
├── runtime/           # Execution context and model wrapper
├── stdlib/            # Standard library (nn, optim, tensor ops)
└── compiler.py        # Main compiler orchestration
```

---

## Standard Library

### `tensor`
Core tensor operations: `zeros`, `ones`, `randn`, `cat`, `stack`, `reshape`, `transpose`, etc.
- `zeros`, `ones`, `randn`, `full` support dynamic shapes via list literals.
- `cat` and `stack` support both variadic arguments `cat(y1, y2, dim=0)` and list literals `cat([y1, y2], dim=0)`.
- `reshape` supports variadic arguments and list literals: `reshape(x, -1, 10, 2)` or `reshape(x, [-1, 10, 2])`.

### `nn`
Neural network layers: `Linear`, `Conv2d`, `LSTM`, `MultiheadAttention`, `ReLU`, `Softmax`, etc.
- `LSTM` and `MultiheadAttention` return tuples and support destructuring: `let (out, hidden) = rnn(x);`.
- `MultiheadAttention` supports 3 inputs: `let (out, weights) = attn(query, key, value);`.

### `dist`
Probability distributions: `Normal`, `Bernoulli`, `Categorical`, `Uniform`, etc.
Support keyword arguments: `Bernoulli(probs=p)`.
Support the `~` operator for observations: `require y ~ Bernoulli(probs=p);`.

### `optim`
Optimizers and schedulers:
- `Adam`, `AdamW`, `SGD`, `RMSprop`, `Adagrad`, `LBFGS`
- Support `lr`, `weight_decay`, and specific hyperparameters (e.g., `betas`, `momentum`).

### `data`
Data loading utilities: `DataLoader`, `Dataset`

---

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_parser.py -v

# Run with coverage
pytest tests/ --cov=delta
```

### Project Status

- [x] Lexer and parser
- [x] Type inference
- [x] SIR (Semantic IR)
- [x] PyTorch FX backend
- [x] Basic constraint compilation
- [x] Mode specialization (train/infer)
- [x] Standard library
- [ ] Probabilistic inference
- [ ] Advanced debugging (`why` command)

---

## Contributing

Contributions are welcome!

1. Fork the repository at [github.com/deltalanguage/delta](https://github.com/deltalanguage/delta)
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Delta builds on ideas from:
- Differentiable programming (JAX, PyTorch)
- Probabilistic programming (Stan, Pyro)
- Constraint-based learning research
- Compiler design (MLIR, XLA)
