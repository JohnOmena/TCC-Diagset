import pandas as pd
from pathlib import Path

# caminhos – ajuste se o seu diretório for diferente
root = Path("E:\DIAGSET\DiagSet-A\partitions")

csvs = [
    root / "DiagSet-A.1" / f"{split}.csv"
    for split in ("train", "validation", "test")
] + [
    root / "DiagSet-A.2" / f"{split}.csv"
    for split in ("train", "validation", "test")
]

for csv in csvs:
    df = pd.read_csv(csv, nrows=5)   # lê só 5 linhas para ser rápido
    print(f"\n{csv.name} ─ {len(df):,} linhas (mostrando 5)")
    print(df.head())