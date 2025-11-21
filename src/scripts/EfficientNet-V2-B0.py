#!/usr/bin/env python3
"""
Treino baseline EfficientNet-V2-B0 · DiagSet-A.1
Com barras tqdm para progresso instantâneo.
"""

import time, torch
from pathlib import Path
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_v2_s
from sklearn.metrics import f1_score
import pandas as pd
import wandb
from tqdm import tqdm

from diagset_data import DiagSetAPatchDataset

# -------- hiperparâmetros embutidos --------
CFG = dict(
    train_csv = "data/splits_A.1/train.csv",
    val_csv   = "data/splits_A.1/validation.csv",
    img_size  = 224,
    batch     = 32,
    lr        = 1e-3,
    epochs    = 10,
    run_name  = "EffNetV2_B0_A1",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

def main():
    # ----- dados -----
    train_df = pd.read_csv(CFG["train_csv"])
    val_df   = pd.read_csv(CFG["val_csv"])

    train_ds = DiagSetAPatchDataset(train_df, train=True,  size=CFG["img_size"])
    val_ds   = DiagSetAPatchDataset(val_df,   train=False, size=CFG["img_size"])

    train_loader = DataLoader(train_ds, batch_size=CFG["batch"], shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch"]*2, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Dataset len: {len(train_ds):,}  |  Batch size: {CFG['batch']}")

    # ----- modelo -----
    model = efficientnet_v2_s(weights="IMAGENET1K_V1")
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 9)
    model.to(DEVICE)

    opt   = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()

    # ----- wandb -----
    wandb.init(project="DiagSetA", name=CFG["run_name"], config=CFG)
    wandb.watch(model, log="all")

    best_f1, patience, no_imp = 0, 3, 0

    for epoch in range(1, CFG["epochs"] + 1):
        model.train(); t0 = time.time()
        prog = tqdm(train_loader, desc=f"Epoch {epoch:02d}", unit="batch")
        for batch in prog:
            imgs  = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            with torch.cuda.amp.autocast():
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()

            prog.set_postfix({"loss": f"{loss.item():.4f}"})
        sched.step()

        # ----- validação -----
        model.eval(); preds, gts = [], []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for batch in val_loader:
                logits = model(batch["image"].to(DEVICE))
                preds.extend(logits.argmax(1).cpu().tolist())
                gts.extend(batch["label"].tolist())
        f1 = f1_score(gts, preds, average="macro")
        wandb.log({"val_macroF1": f1, "epoch": epoch})

        print(f"\nEpoch {epoch:02d}  F1={f1:.4f}  "
              f"LR={opt.param_groups[0]['lr']:.2e}  "
              f"Time={time.time()-t0:.1f}s")

        if f1 > best_f1 + 1e-4:
            best_f1, no_imp = f1, 0
            torch.save(model.state_dict(), "ckpt_effnetA1_best.pt")
        else:
            no_imp += 1
            if no_imp >= patience:
                print("Early stopped."); break

    wandb.finish()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
