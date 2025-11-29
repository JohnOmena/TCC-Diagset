from pathlib import Path
import pandas as pd

# Prefixos antigo e novo — ATENÇÃO nas barras
OLD_PREFIX = r"E:\DIAGSET"
NEW_PREFIX = r"D:\DIAGSET"

ROOT = Path(r"D:\DIAG\src\data")


def fix_split_dir(split_dir: Path):
    for name in ["train.csv", "validation.csv", "test.csv"]:
        csv_path = split_dir / name
        print(f"[INFO] Ajustando {csv_path}")
        df = pd.read_csv(csv_path)

        # Substitui só o começo do path (sem regex, substituição literal)
        df["path"] = df["path"].str.replace(OLD_PREFIX, NEW_PREFIX, regex=False)

        df.to_csv(csv_path, index=False)


def main():
    for mag in ["5x", "10x", "20x"]:
        split_dir = ROOT / f"splits_A.2_{mag}"
        fix_split_dir(split_dir)


if __name__ == "__main__":
    main()
