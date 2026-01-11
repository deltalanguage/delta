#!/usr/bin/env python3
"""Tiny Transformer with Delta - Minimal Example"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader
import delta

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

TEXT = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
"""

class TextData(Dataset):
    def __init__(self, text, ctx=32):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.ctx = ctx
    def __len__(self): return len(self.data) - self.ctx
    def __getitem__(self, i): return self.data[i:i+self.ctx], self.data[i+1:i+self.ctx+1]

# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────

print("═" * 50)
print("  Tiny Transformer with Delta")
print("═" * 50)

data = TextData(TEXT)
loader = DataLoader(data, batch_size=16, shuffle=True)

model = delta.compile("model.delta")
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.set_vocab(TEXT)
model.fit(loader, epochs=100, lr=3e-3)

# ─────────────────────────────────────────────────────────────────────────────
# Generate
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═" * 50)
print("  Generated Text")
print("═" * 50)
print(model.generate("First Citizen:\n", max_tokens=200))
