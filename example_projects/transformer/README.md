# Tiny Transformer

A minimal character-level GPT trained on Shakespeare text.

## Architecture

- **Vocab**: 46 characters (Shakespeare subset)
- **Embedding**: 64 dimensions
- **Attention**: 2 heads
- **Context**: 32 tokens
- **FFN**: 256 hidden units
- **Layers**: 1 transformer block

## Files

- `model.delta` - Delta language specification of the transformer
- `train.py` - Training script (currently uses PyTorch directly)

## Run

```bash
cd example_projects/transformer
python train.py
```

## Output

```
══════════════════════════════════════════════════
  Tiny Transformer
══════════════════════════════════════════════════
  Vocab: 46 chars
  Text: 1000 chars
══════════════════════════════════════════════════

Training on cpu...
──────────────────────────────────────────────────
Epoch  10  │  Loss: 1.2240
Epoch  20  │  Loss: 0.2699
...
Epoch 100  │  Loss: 0.1528
──────────────────────────────────────────────────

══════════════════════════════════════════════════
  Generated Text
══════════════════════════════════════════════════
First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.
...
```

## Delta Language

The `model.delta` file shows the intended Delta syntax for transformers:

```delta
// Self-attention with causal masking
fn attention(x: Tensor) -> Tensor {
    let q = matmul(x, W_q);
    let k = matmul(x, W_k);
    let v = matmul(x, W_v);
    
    let scores = matmul(q, transpose(k, -2, -1)) / sqrt(64.0);
    let masked = scores + causal_mask(scores);
    let attn = softmax(masked, dim=-1);
    
    matmul(matmul(attn, v), W_o)
}

// Training block with optimizer config
learn train {
    let logits = forward(tokens);
    cross_entropy(logits, targets)
} with optimizer = AdamW(lr=3e-3), epochs = 100
```

The Delta compiler supports simpler models like the MNIST feedforward network.
Full transformer support is planned for a future release.
