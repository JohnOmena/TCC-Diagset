import random, json
import numpy as np
from pathlib import Path

BLOBS = list(Path("E:/DIAGSET/DiagSet-A/blobs/S/10x").rglob("*.npy"))
random.shuffle(BLOBS)

shapes = {}
for p in BLOBS[:30]:
    arr = np.load(p, mmap_mode="r")
    shapes.setdefault(arr.ndim, set()).add(arr.shape)
    print(p.name, arr.shape)

print("\nResumo por ndim:")
print(json.dumps({k: list(map(str, v)) for k, v in shapes.items()},
                 indent=2))
