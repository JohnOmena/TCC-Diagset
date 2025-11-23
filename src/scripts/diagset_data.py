#!/usr/bin/env python3
"""
diagset_data.py
---------------
Dataset utilitário para a base DIAGSET-A (pipeline do artigo):

• Patches 256×256 lidos de blobs .npy (mem-map)
• Augmentations:
    - RandomCrop 224 (treino) / CenterCrop 224 (val/test)
    - Horizontal Flip
    - Rotação {0°, 90°, 180°, 270°}
• Dois modos de rótulo:
      mode="s5" → 9 classes  (BG, T, N, A, R1–R5)
      mode="s1" → binário    (não-câncer × câncer),
                   com mapeamento alinhado ao paper:
                   0 = {BG, T, N, A}
                   1 = {R1, R2, R3, R4, R5}
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Literal

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

# ---------- mapeamentos ----------
# S5: 9 classes separadas, na ordem lógica do artigo:
# BG, T, N, A, R1, R2, R3, R4, R5
LABEL2IDX_S5: Dict[str, int] = {
    "BG": 0,
    "T":  1,
    "N":  2,
    "A":  3,
    "R1": 4,
    "R2": 5,
    "R3": 6,
    "R4": 7,
    "R5": 8,
}


def label_to_idx(lbl: str, mode: Literal["s5", "s1"]) -> int:
    """
    Converte o rótulo textual (A/BG/N/T/R1–R5) para índice numérico.

    mode = "s5":
        9 classes (BG, T, N, A, R1–R5)

    mode = "s1":
        binário, alinhado ao artigo DiagSet:
            0 = não-câncer → {BG, T, N, A}
            1 = câncer     → {R1, R2, R3, R4, R5}
    """
    if mode == "s5":
        return LABEL2IDX_S5[lbl]

    # S1 → não-câncer vs câncer (mapeamento do paper)
    return 0 if lbl in {"BG", "T", "N", "A"} else 1


# ---------- transforms ----------
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)


class RandomRotate90:
    """Rotaciona a imagem por múltiplos de 90° (compatível com qualquer versão do torchvision)."""
    def __call__(self, img: Image.Image) -> Image.Image:
        k = np.random.randint(0, 4)   # 0,1,2,3
        return img.rotate(90 * k)

    def __repr__(self):              # para aparecer no print(transform)
        return f"{self.__class__.__name__}()"


def _build_transforms(train: bool, crop_size: int):
    if train:
        return T.Compose([
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(),
            RandomRotate90(),
            T.ToTensor(),
            T.Normalize(IMNET_MEAN, IMNET_STD),
        ])
    else:
        return T.Compose([
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(IMNET_MEAN, IMNET_STD),
        ])


# ---------- Dataset ----------
class DiagSetAPatchDataset(Dataset):
    """
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Must contain: path, patch_idx, slide_id, mag, label
    mode : 's5' or 's1'
    crop_size : int (default 224)
    cache_blobs : bool
        True → mantém mem-maps em cache (↑RAM, ↓I/O)
    """

    def __init__(
        self,
        dataframe,
        train: bool = True,
        mode: Literal["s5", "s1"] = "s5",
        crop_size: int = 224,
        cache_blobs: bool = True,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.mode = mode
        self.tfm = _build_transforms(train, crop_size)
        self.cache: Optional[Dict[Path, np.memmap]] = {} if cache_blobs else None

    # ---- helpers ----
    def _get_mm(self, path: Path) -> np.memmap:
        if self.cache is None:
            return np.load(path, mmap_mode="r")
        mm = self.cache.get(path)
        if mm is None:
            mm = np.load(path, mmap_mode="r")
            self.cache[path] = mm
        return mm

    def _read_patch(self, row):
        mm = self._get_mm(Path(row.path))
        return mm[int(row.patch_idx)]        # (256,256,3) uint8

    # ---- Dataset ----
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_np = self._read_patch(row)
        img = self.tfm(Image.fromarray(img_np))
        label_idx = label_to_idx(row.label, self.mode)
        return {
            "image": img,
            "label": torch.tensor(label_idx, dtype=torch.long),
        }