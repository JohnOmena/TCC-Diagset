import pandas as pd
from pathlib import Path

for mag in ["5x", "10x", "20x"]:
    csv_path = Path("data") / f"splits_A.2_{mag}" / "test.csv"
    df = pd.read_csv(csv_path, nrows=5)  # só 5 linhas pra ser rápido
    print(f"\n=== {mag} ===")
    print(df["path"])
    print("prefixo ok?:", df["path"].str.startswith(r"D:\DIAGSET").all())
