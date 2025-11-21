#!/usr/bin/env python3
"""
build_patch_index_blobs.py
--------------------------
Cria um índice *por patch* a partir dos blobs .npy do DiagSet-A.

Saída: CSV com colunas
    path, patch_idx, slide_id, mag, label
"""

from __future__ import annotations
import csv, sys, re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Iterable

class CONFIG:
    ROOT         = Path("E:/DIAGSET/DiagSet-A")     # raiz da base
    MAGNIF       = ["5x", "10x", "20x", "40x"]                 # ampliações desejadas
    OUT_CSV = Path("data/patch_index_blobs_4mag.csv")

CLASSES: Tuple[str, ...] = ("A","BG","N","R1","R2","R3","R4","R5","T")
BLOB_RE = re.compile(r"\.blob\.(\d+)\.npy$")        # para extrair sub-id (opcional)

def iter_blobs(root: Path, mags: Iterable[str]):
    """Rende (blob_path, slide_id, mag, label)"""
    for mag in mags:
        mag_dir = root / "blobs" / "S" / mag
        if not mag_dir.exists():
            print(f"[WARN] '{mag}' não existe", file=sys.stderr); continue
        for slide_dir in mag_dir.iterdir():
            if not slide_dir.is_dir(): continue
            slide_id = slide_dir.name
            for label in CLASSES:
                for blob in (slide_dir / label).glob("*.npy"):
                    yield blob, slide_id, mag, label

def build_csv():
    CONFIG.OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with CONFIG.OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","patch_idx","slide_id","mag","label"])
        for blob, slide_id, mag, label in iter_blobs(CONFIG.ROOT, CONFIG.MAGNIF):
            try:
                arr = np.load(blob, mmap_mode="r")
            except Exception as e:
                print(f"[ERRO] {blob}: {e}", file=sys.stderr); continue
            if arr.ndim != 4 or arr.shape[1:] != (256,256,3):
                print(f"[SKIP] {blob} shape={arr.shape}", file=sys.stderr); continue
            for idx in range(arr.shape[0]):
                w.writerow([str(blob), idx, slide_id, mag, label])
            total += arr.shape[0]
    print(f"[OK] Indexados {total:,} patches → {CONFIG.OUT_CSV}")

if __name__ == "__main__":
    build_csv()
