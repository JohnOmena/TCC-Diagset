#!/usr/bin/env python3

# ────────── Filtros de avisos ─────────────────────────────────────────────
import warnings, re, io
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=re.escape("Detected call of `lr_scheduler.step()` before `optimizer.step()`"),
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*weights_only=False.*",
)

# ────────── Sementes e determinismo ───────────────────────────────────────
import random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────────────────────────────────────────
from pathlib import Path
import os, csv, time, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_v2_s
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import wandb, matplotlib.pyplot as plt, seaborn as sns
from diagset_data import DiagSetAPatchDataset

# --------------------------- CONFIG ---------------------------------------
class CONFIG:
    MAG          = "5x"
    MODE         = "s5"
    IMG_SIZE     = 256
    BATCH        = 24
    EPOCHS       = 20
    PATIENCE     = 10
    LR           = 1e-3
    NUM_WORKERS  = 4
    WEIGHT_DECAY = 1e-4
    FP16         = True
    PROJECT      = "DiagSetA"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------- helpers ----------------------------------------------------
def evaluate(model, loader, fp16=False):
    model.eval(); preds, gts, losses = [], [], []
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad(), torch.autocast("cuda", enabled=fp16):
        for b in loader:
            x, y = b["image"].to(device), b["label"].to(device)
            log = model(x)
            losses.extend(ce(log, y).cpu().tolist())
            preds.extend(log.argmax(1).cpu()); gts.extend(y.cpu())
    return float(np.mean(losses)), accuracy_score(gts, preds), \
           f1_score(gts, preds, average="macro"), preds, gts

def grad_global_norm(model):
    return torch.sqrt(sum((p.grad.norm(2) ** 2
                           for p in model.parameters() if p.grad is not None))).item()

# --------------------------- MAIN -----------------------------------------
def main():
    run_name = f"EffNet_{CONFIG.MAG}_{CONFIG.MODE}"

    ckpt_dir  = Path("checkpoints") / CONFIG.MAG
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / f"ckpt_{run_name}_last.pt"
    best_ckpt = ckpt_dir / f"ckpt_{run_name}_best.pt"

    tr_df = pd.read_csv(f"data/splits_A.1_{CONFIG.MAG}/train.csv")
    vl_df = pd.read_csv(f"data/splits_A.1_{CONFIG.MAG}/validation.csv")

    tr_ds = DiagSetAPatchDataset(tr_df, train=True, mode=CONFIG.MODE, size=CONFIG.IMG_SIZE)
    vl_ds = DiagSetAPatchDataset(vl_df, train=False, mode=CONFIG.MODE, size=CONFIG.IMG_SIZE)

    tr_dl = DataLoader(tr_ds, batch_size=CONFIG.BATCH, shuffle=True,
                       num_workers=CONFIG.NUM_WORKERS, pin_memory=True, persistent_workers=True)
    vl_dl = DataLoader(vl_ds, batch_size=CONFIG.BATCH*2, shuffle=False, num_workers=0, pin_memory=False)

    samp_idx = np.random.choice(len(tr_ds), min(20_000, len(tr_ds)), False)
    tr_eval_dl = DataLoader(torch.utils.data.Subset(tr_ds, samp_idx),
                            batch_size=CONFIG.BATCH*2, shuffle=False, num_workers=0, pin_memory=False)

    steps_ep = len(tr_dl)

    model = efficientnet_v2_s(weights="IMAGENET1K_V1")
    model.classifier[-1].out_features = 9 if CONFIG.MODE == "s5" else 2
    model.to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=CONFIG.LR, weight_decay=CONFIG.WEIGHT_DECAY)
    sched = OneCycleLR(opt, max_lr=CONFIG.LR, total_steps=CONFIG.EPOCHS*steps_ep,
                       pct_start=0.3, div_factor=25, final_div_factor=1e4, last_epoch=-1)
    scaler = torch.amp.GradScaler(enabled=CONFIG.FP16)
    ce_loss = torch.nn.CrossEntropyLoss()

    start_epoch = 1
    if last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        wandb_id = ckpt["wandb_id"]
        print(f"[INFO] Resumindo da época {start_epoch}")
    else:
        wandb_id = None

    wandb.init(project=CONFIG.PROJECT, name=run_name,
               id=wandb_id, resume="allow" if wandb_id else None,
               config={k:v for k,v in CONFIG.__dict__.items() if k.isupper()})
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    wandb.watch(model, log="gradients", log_freq=500)

    os.makedirs("logs", exist_ok=True)
    csv_path = Path("logs") / f"{run_name}.csv"
    new_csv = not csv_path.exists()
    csvf = open(csv_path, "a", newline=""); wr = csv.writer(csvf)
    if new_csv:
        wr.writerow(["epoch","train_loss","train_acc","train_f1",
                     "val_loss","val_acc","val_f1",
                     "grad_norm","batch_time","gpu_mem","lr"])

    best_f1, no_imp = 0, 0

    for epoch in range(start_epoch, CONFIG.EPOCHS+1):
        model.train()
        run_loss, run_grad, btimes = 0., 0., []

        prog = tqdm(tr_dl, desc=f"Epoch {epoch:02d}", leave=False)
        for b in prog:
            t0 = time.time()
            x = b["image"].to(device, non_blocking=True)
            y = b["label"].to(device)

            with torch.autocast("cuda", enabled=CONFIG.FP16):
                loss = ce_loss(model(x), y)

            scaler.scale(loss).backward()
            run_grad += grad_global_norm(model)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            sched.step()

            run_loss += loss.item()
            btimes.append(time.time()-t0)
            prog.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = run_loss / len(tr_dl)
        grad_norm  = run_grad / len(tr_dl)
        batch_time = float(np.mean(btimes))
        gpu_mem = torch.cuda.max_memory_allocated()/1024**2
        torch.cuda.reset_peak_memory_stats()

        _, tr_acc, tr_f1, _, _ = evaluate(model, tr_eval_dl, CONFIG.FP16)
        val_loss, val_acc, val_f1, preds_v, gts_v = evaluate(model, vl_dl, CONFIG.FP16)

        if epoch % 5 == 0:
            n_cls = 9 if CONFIG.MODE=="s5" else 2
            cm = confusion_matrix(gts_v, preds_v, labels=range(n_cls))
            fig, ax = plt.subplots(figsize=(5,5))
            sns.heatmap(cm, ax=ax, cmap="Blues", cbar=False,
                        annot=True, fmt="d",
                        xticklabels=range(n_cls),
                        yticklabels=range(n_cls))
            ax.set_xlabel("Pred"); ax.set_ylabel("True")
            wandb.log({"confmat": wandb.Image(fig)}, step=epoch)
            plt.close(fig)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": tr_acc,
            "train_macroF1": tr_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macroF1": val_f1,
            "grad_norm": grad_norm,
            "batch_time": batch_time,
            "gpu_mem_MB": gpu_mem,
            "lr": sched.get_last_lr()[0],
        })

        wr.writerow([epoch,f"{train_loss:.5f}",f"{tr_acc:.4f}",f"{tr_f1:.4f}",
                     f"{val_loss:.5f}",f"{val_acc:.4f}",f"{val_f1:.4f}",
                     f"{grad_norm:.2f}",f"{batch_time:.3f}",
                     f"{gpu_mem:.1f}",f"{sched.get_last_lr()[0]:.6f}"])
        csvf.flush()

        print(f"Epoch {epoch:02d} train_f1={tr_f1:.3f} val_f1={val_f1:.3f} Δ={tr_f1-val_f1:+.3f}")

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "wandb_id": wandb.run.id,
            }, last_ckpt)

        if val_f1 > best_f1 + 1e-4:
            best_f1, no_imp = val_f1, 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_imp += 1
            if no_imp >= CONFIG.PATIENCE:
                print("Early stopped."); break

    csvf.close(); wandb.finish()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
