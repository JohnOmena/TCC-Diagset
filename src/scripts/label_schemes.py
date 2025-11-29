#!/usr/bin/env python3
"""
label_schemes.py
----------------
Utilitários para colapsar rótulos S5 (9 classes) nos esquemas S1–S4,
seguindo exatamente a Tabela 10 do artigo DiagSet.

Ordem das classes em S5 (índices 0–8), compatível com diagset_data.LABEL2IDX_S5:

    0: BG
    1: T
    2: N
    3: A
    4: R1
    5: R2
    6: R3
    7: R4
    8: R5

Os arrays abaixo mapeiam cada índice de S5 para um grupo 0-based em S1–S4.
"""

from __future__ import annotations
from typing import Sequence, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[int]]

# Índices só por clareza (iguais aos do LABEL2IDX_S5 em diagset_data.py)
IDX_BG = 0
IDX_T  = 1
IDX_N  = 2
IDX_A  = 3
IDX_R1 = 4
IDX_R2 = 5
IDX_R3 = 6
IDX_R4 = 7
IDX_R5 = 8

# -----------------------------
# Mapas de colapso (0-based)
# -----------------------------
# S1 (2 grupos):
#   grupo 0 → {BG, T, N, A}
#   grupo 1 → {R1, R2, R3, R4, R5}
S1_MAP = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)

# S2 (2 grupos):
#   grupo 0 → {BG, T, N, A, R1, R2}
#   grupo 1 → {R3, R4, R5}
S2_MAP = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int64)

# S3 (4 grupos):
#   grupo 0 → {BG, T, N, A, R1, R2}
#   grupo 1 → {R3}
#   grupo 2 → {R4}
#   grupo 3 → {R5}
S3_MAP = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3], dtype=np.int64)

# S4 (6 grupos):
#   grupo 0 → {BG, T, N, A}
#   grupo 1 → {R1}
#   grupo 2 → {R2}
#   grupo 3 → {R3}
#   grupo 4 → {R4}
#   grupo 5 → {R5}
S4_MAP = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5], dtype=np.int64)

# S5 (9 grupos, identidade)
S5_MAP = np.arange(9, dtype=np.int64)


def _collapse_labels(labels: ArrayLike, mapping: np.ndarray):
    """
    Aplica o mapeamento de colapso em um array de rótulos S5 (0..8).

    labels: pode ser
        - lista de ints
        - numpy.ndarray de ints
        - torch.Tensor de ints (qualquer shape, contanto que os valores sejam 0..8)
    """
    if isinstance(labels, torch.Tensor):
        mapping_t = torch.as_tensor(mapping, dtype=torch.long, device=labels.device)
        return mapping_t[labels]
    else:
        arr = np.asarray(labels, dtype=np.int64)
        return mapping[arr]


def collapse_to_S1(labels: ArrayLike):
    """Colapsa rótulos S5 para S1 (2 classes)."""
    return _collapse_labels(labels, S1_MAP)


def collapse_to_S2(labels: ArrayLike):
    """Colapsa rótulos S5 para S2 (2 classes)."""
    return _collapse_labels(labels, S2_MAP)


def collapse_to_S3(labels: ArrayLike):
    """Colapsa rótulos S5 para S3 (4 classes)."""
    return _collapse_labels(labels, S3_MAP)


def collapse_to_S4(labels: ArrayLike):
    """Colapsa rótulos S5 para S4 (6 classes)."""
    return _collapse_labels(labels, S4_MAP)


def collapse_to_S5(labels: ArrayLike):
    """Identidade (mantém S5); útil só para consistência de API."""
    return _collapse_labels(labels, S5_MAP)


__all__ = [
    "collapse_to_S1",
    "collapse_to_S2",
    "collapse_to_S3",
    "collapse_to_S4",
    "collapse_to_S5",
    "S1_MAP",
    "S2_MAP",
    "S3_MAP",
    "S4_MAP",
    "S5_MAP",
]
