import pandas as pd
from torch.utils.data import DataLoader
from diagset_data import DiagSetAPatchDataset

confs = [("5x", 224, 24), ("10x", 224, 24), ("20x", 224, 24), ("40x", 224, 20)]

for mag, img, bs in confs:
    df = pd.read_csv(f"data/splits_A.1_{mag}/train.csv").sample(bs, random_state=42)
    ds = DiagSetAPatchDataset(df, train=True, crop_size=img)
    dl = DataLoader(ds, batch_size=bs, num_workers=0)
    batch = next(iter(dl))
    print(f"{mag} ✔︎ {batch['image'].shape}")