from pathlib import Path
import pandas as pd

PATCH_INDEX   = Path("data/patch_index_blobs_4mag.csv")
PARTITIONS    = Path("E:\DIAGSET\DiagSet-A\partitions")          # DiagSet-A.1 / A.2
OUT_ROOT      = Path("E:\DIAGSET\DiagSet-A\src\data")                # onde salvar splits

index_df = pd.read_csv(PATCH_INDEX)

for mag in ["5x", "10x", "20x", "40x"]:
    mag_idx = index_df[index_df["mag"] == mag]
    for ver in ("A.1", "A.2"):
        out_dir = OUT_ROOT / f"splits_{ver}_{mag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation", "test"):
            slide_ids = pd.read_csv(
                PARTITIONS / f"DiagSet-{ver}/{split}.csv"
            )["scan_id"]
            df_split = mag_idx[mag_idx["slide_id"].isin(slide_ids)]
            df_split.to_csv(out_dir / f"{split}.csv", index=False)
            print(f"[{ver}-{mag}] {split}: {len(df_split):,} patches")