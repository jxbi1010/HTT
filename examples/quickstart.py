"""Minimal end-to-end example: raw tactile reading -> [1, 192] embedding.

Run from the repo root:
    python examples/quickstart.py
"""

import sys
from pathlib import Path

import numpy as np

# Make the repo root importable when running from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from htt import HTT

# --- Vision example (gsmini / 9dtact) ------------------------------------
# A real reading is a uint8 RGB frame [224, 224, 3] from the sensor.
tf_vision = HTT(modality="gsmini")
raw_frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)  # ← your frame
emb = tf_vision(raw_frame)
print(f"gsmini embedding: {tuple(emb.shape)}  norm={emb.norm().item():.3f}")

# --- Taxel example (xela / tac02) ----------------------------------------
# A real reading is a float array [T, 72] (xela) or [T, 66] (tac02).
tf_taxel = HTT(modality="xela")
raw_taxel = np.random.randn(20, 72).astype(np.float32)               # ← your reading
emb = tf_taxel(raw_taxel)
print(f"xela   embedding: {tuple(emb.shape)}  norm={emb.norm().item():.3f}")

# The embedding is the mean-pooled shared-trunk feature — feed it to any head.
