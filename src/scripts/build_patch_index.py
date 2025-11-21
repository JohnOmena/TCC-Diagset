#!/usr/bin/env python3
"""
build_patch_index.py  –  v2  (somente arquivos-imagem)
"""

from __future__ import annotations
import csv, sys
from pathlib import Path
from typing import Iterable, List, Tuple

class CONFIG:
    ROOT: Path = Path("E:/DIAGSET/DiagSet-A")
    MAGNIFICATIONS: List[str] = ["10x", "20x"]
    OUT_CSV: Path = Path("data/patch_index_img.csv")

CLASSES: Tuple[str, ...] = ("A","BG","N","R1","R2","R3","R4","R5","T")
VALID_EXTS: Tuple[str, ...] = (".npy",)   # só .npy mesmo
CONFIG.OUT_CSV = Path("data/patch_index.npy.csv")

def iter_patch_paths(root: Path, mags: Iterable[str]):
    blobs_root = root / "blobs" / "S"
    if not blobs_root.exists():
        sys.exit(f"ERRO: {blobs_root} não existe")
    for mag in mags:
        for slide_dir in (blobs_root / mag).glob("*"):
            if not slide_dir.is_dir(): continue
            slide_id = slide_dir.name
            for label in CLASSES:
                for patch_path in (slide_dir / label).rglob("*"):
                    if patch_path.suffix.lower() in VALID_EXTS:
                        yield patch_path, slide_id, mag, label

def build_csv():
    CONFIG.OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with CONFIG.OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["path","slide_id","mag","label"])
        for p,s,m,l in iter_patch_paths(CONFIG.ROOT, CONFIG.MAGNIFICATIONS):
            w.writerow([str(p),s,m,l]); total += 1
    print(f"[OK] {total:,} patches → {CONFIG.OUT_CSV}")

if __name__ == "__main__":
    build_csv()