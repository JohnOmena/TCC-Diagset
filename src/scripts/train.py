#!/usr/bin/env python3
"""
train.py · DIAGSET-A benchmark (batch-accum + W&B resume embutido)

Como retomar uma run:
---------------------
1) abra a run no W&B, clique em ⚙️ → Copy ID (ex.: ms7gijc9)
2) edite o bloco RESUME SETTINGS:
       RESUME_RUN_ID    = "ms7gijc9"
       RESUME_STRATEGY  = "must"     # "allow" ou "must"
       DOWNLOAD_CHECKPT = True       # baixa ckpt_last_* se não existir
3) `python train.py`
"""

from __future__ import annotations

import io
import os, random, time, csv
from math import ceil
from pathlib import Path

import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import wandb, seaborn as sns, matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image                                            # ← NEW
from diagset_data import DiagSetAPatchDataset


# ╭───────────────────  RESUME SETTINGS  ───────────────────╮
RESUME_RUN_ID    = "cdiopf59"   # "" ou None ⇒ começa run nova
RESUME_STRATEGY  = "must"       # "allow" | "must" | None
DOWNLOAD_CHECKPT = True         # baixa ckpt_last_* da nuvem se faltar local
# ╰──────────────────────────────────────────────────────────╯


# ---------- W&B sem serviço ----------
os.environ["WANDB_DISABLE_SERVICE"] = "true"
os.environ["WANDB_START_METHOD"]    = "thread"
# -------------------------------------


# ---------------- CONFIG -------------
class CONFIG:
    model   = "convnext"             # "effnet" | "convnext" | "swin" | "deit"
    mag     = "20x"              # "5x" | "10x" | "20x" | "40x"
    schema  = "s1"               # "s5" | "s1"
    split   = "A.2"              # "A.1" | "A.2"

    GPU_TAG = "RTX4060"          # "RTX3060", "GTX1060", "RTX4060"
    _PRESETS = {
        "RTX3060": dict(batch_gpu=32, num_workers=8),
        "GTX1060": dict(batch_gpu=16, num_workers=4),
        "RTX4060": dict(batch_gpu=32, num_workers=6),
    }

    seed            = 42
    img_size        = 224
    batch_gpu       = 16           # sobrescrito pelo preset
    batch_effective = 32
    epochs          = 50
    lr              = 1e-4
    weight_decay    = 5e-4
    num_workers     = 8
    step_milestone  = (20, 40)
    early_patience  = 50

    project      = "DiagSetA"
    log_grad     = False
    wandb_mode   = "online"        # "online" | "offline" | "disabled"
    resume       = True
    resume_best  = False
    ckpt_override_path: str | None = None


for k, v in CONFIG._PRESETS[CONFIG.GPU_TAG].items():
    setattr(CONFIG, k, v)


# ---------- bloco de resume automático ------------------
WANDB_KWARGS = {}
if RESUME_RUN_ID and RESUME_STRATEGY:
    WANDB_KWARGS.update(dict(id=RESUME_RUN_ID, resume=RESUME_STRATEGY))
    os.environ["WANDB_RUN_ID"] = RESUME_RUN_ID
    os.environ["WANDB_RESUME"] = RESUME_STRATEGY

    if DOWNLOAD_CHECKPT:
        ckpt_dir = Path("checkpoints") / CONFIG.mag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_last = ckpt_dir / f"ckpt_last_{CONFIG.model}_{CONFIG.schema}.pt"
        if not ckpt_last.exists():
            print(f"[INFO] {ckpt_last.name} não encontrado, baixando da nuvem…")
            api = wandb.Api()
            run = api.run(f"/{CONFIG.project}/runs/{RESUME_RUN_ID}")
            run.file(ckpt_last.name).download(root=str(ckpt_dir), replace=True)
            print("[INFO] checkpoint baixado.")
else:
    os.environ.pop("WANDB_RUN_ID", None)
    os.environ.pop("WANDB_RESUME", None)


# ------------- utils -----------------
def seed_everything(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)

def grad_global_norm(m):
    return torch.sqrt(sum(p.grad.norm(2) ** 2
                          for p in m.parameters() if p.grad is not None)).item()
# -------------------------------------


def get_model(name: str, n_cls: int):
    import torch.nn as nn
    name = name.lower()

    if name == "effnet":
        from torchvision.models import efficientnet_v2_s
        m = efficientnet_v2_s(weights="IMAGENET1K_V1")
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_cls)
    elif name == "convnext":
        from torchvision.models import convnext_tiny
        m = convnext_tiny(weights="IMAGENET1K_V1")
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_cls)
    elif name == "swin":
        import timm
        m = timm.create_model("swin_tiny_patch4_window7_224",
                              pretrained=True, num_classes=n_cls)
    elif name == "deit":
        import timm
        m = timm.create_model("deit_small_patch16_224",
                              pretrained=True, num_classes=n_cls)
    else:
        raise ValueError("Modelo inválido")
    return m


# ------------- MAIN ------------------
def main(cfg: CONFIG):

    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accum_steps = ceil(cfg.batch_effective / cfg.batch_gpu)

    split_dir = Path(f"data/splits_{cfg.split}_{cfg.mag}")
    train_df  = pd.read_csv(split_dir / "train.csv")
    val_df    = pd.read_csv(split_dir / "validation.csv")

    train_ds = DiagSetAPatchDataset(train_df, True,
                                    mode=cfg.schema, crop_size=cfg.img_size)
    val_ds   = DiagSetAPatchDataset(val_df, False,
                                    mode=cfg.schema, crop_size=cfg.img_size)

    train_dl = DataLoader(train_ds, cfg.batch_gpu, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True,
                          persistent_workers=(cfg.num_workers > 0),
                          worker_init_fn=seed_worker)
    val_dl   = DataLoader(val_ds, cfg.batch_gpu * 2, shuffle=False,
                          num_workers=0, pin_memory=False)

    n_cls = 9 if cfg.schema == "s5" else 2
    model = get_model(cfg.model, n_cls).to(device)

    opt   = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9,
                            weight_decay=cfg.weight_decay)
    sched = StepLR(opt, step_size=cfg.step_milestone[0], gamma=0.1)
    second_step = cfg.step_milestone[1]
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    ckpt_dir  = Path("checkpoints") / cfg.mag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last = ckpt_dir / f"ckpt_last_{cfg.model}_{cfg.schema}.pt"
    ckpt_best = ckpt_dir / f"ckpt_best_{cfg.model}_{cfg.schema}.pt"

    start_epoch = 1
    best_f1, wait = 0., 0

    ckpt_load_path = None
    if cfg.ckpt_override_path:
        ckpt_load_path = Path(cfg.ckpt_override_path)
    elif cfg.resume_best and ckpt_best.exists():
        ckpt_load_path = ckpt_best
    elif cfg.resume and ckpt_last.exists():
        ckpt_load_path = ckpt_last

    if ckpt_load_path and ckpt_load_path.exists():
        ckpt = torch.load(ckpt_load_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["scheduler"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            best_f1     = ckpt.get("best_f1", 0.)
            wait        = ckpt.get("wait", 0)
        else:
            model.load_state_dict(ckpt)
        print(f"[INFO] Retomando do epoch {start_epoch} (ckpt: {ckpt_load_path.name})")
    else:
        print("[INFO] Treino iniciado do zero")

    run_name = f"{cfg.model}_{cfg.mag}_{cfg.schema}_{cfg.split}_{cfg.GPU_TAG}"
    csv_path = Path("logs") / f"{run_name}.csv"
    csv_path.parent.mkdir(exist_ok=True)
    new_csv = not csv_path.exists()
    csvf = open(csv_path, "a", newline="")
    wr = csv.writer(csvf)
    if new_csv:
        wr.writerow(["epoch","train_loss","train_acc","train_f1",
                     "val_loss","val_acc","val_f1",
                     "grad_norm","batch_time","gpu_mem","lr"])

    cfg_dict = {k: getattr(cfg, k) for k in dir(cfg)
                if not k.startswith("_") and not callable(getattr(cfg, k))}

    wandb.init(project=cfg.project, name=run_name, config=cfg_dict,
               mode=cfg.wandb_mode,
               settings=wandb.Settings(start_method="thread"),
               **WANDB_KWARGS)

    if cfg.log_grad:
        wandb.watch(model, log="gradients", log_freq=500)

    for epoch in range(start_epoch, cfg.epochs + 1):

        # ----- treino -----
        model.train()
        tr_preds, tr_gts = [], []
        run_loss = run_grad = 0.
        btimes, steps = [], 0
        t0_epoch = time.time()

        prog = tqdm(train_dl, desc=f"Epoch {epoch:02d}", leave=False)
        for i, b in enumerate(prog, 1):
            t0_batch = time.time()
            x, y = b["image"].to(device, non_blocking=True), b["label"].to(device)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                out  = model(x)
                loss = criterion(out, y) / accum_steps
            scaler.scale(loss).backward()

            tr_preds.extend(out.argmax(1).cpu()); tr_gts.extend(y.cpu())
            run_loss += loss.item() * accum_steps
            btimes.append(time.time() - t0_batch)

            if i % accum_steps == 0 or i == len(train_dl):
                run_grad += grad_global_norm(model)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                steps += 1
            prog.set_postfix(loss=f"{loss.item() * accum_steps:.4f}")

        if epoch == second_step:
            for g in opt.param_groups: g["lr"] *= 0.1
        else:
            sched.step()

        grad_norm  = run_grad / steps
        batch_time = float(np.mean(btimes))
        gpu_mem    = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()

        tr_loss = run_loss / len(train_dl)
        tr_f1   = f1_score(tr_gts, tr_preds, average="macro")
        tr_acc  = accuracy_score(tr_gts, tr_preds)

        # ----- validação -----
        model.eval(); preds, gts, val_loss = [], [], 0.
        with torch.no_grad(), torch.autocast(device_type=device, enabled=(device == "cuda")):
            for b in val_dl:
                out = model(b["image"].to(device))
                preds.extend(out.argmax(1).cpu()); gts.extend(b["label"])
                val_loss += criterion(out, b["label"].to(device)).item()
        val_loss /= len(val_dl)
        val_f1  = f1_score(gts, preds, average="macro")
        val_acc = accuracy_score(gts, preds)

        # matriz de confusão a cada 5 épocas — versão em memória
        if epoch % 5 == 0:
            cm = confusion_matrix(gts, preds, labels=list(range(n_cls)))
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=range(n_cls), yticklabels=range(n_cls), ax=ax)
            ax.set_xlabel("Pred"); ax.set_ylabel("True")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            pil_img = Image.open(buf)                               # ← NEW
            wandb.log({"confusion_matrix": wandb.Image(pil_img)}, step=epoch)
            plt.close(fig)

        # wandb + CSV
        wandb.log({"epoch": epoch,
                   "train_loss": tr_loss, "train_acc": tr_acc, "train_macroF1": tr_f1,
                   "val_loss": val_loss,   "val_acc": val_acc,   "val_macroF1": val_f1,
                   "grad_norm": grad_norm, "batch_time": batch_time,
                   "gpu_mem_MB": gpu_mem,  "lr": opt.param_groups[0]["lr"],
                   "time_epoch_s": time.time() - t0_epoch},
                   step=epoch)

        wr.writerow([epoch,f"{tr_loss:.5f}",f"{tr_acc:.4f}",f"{tr_f1:.4f}",
                     f"{val_loss:.5f}",f"{val_acc:.4f}",f"{val_f1:.4f}",
                     f"{grad_norm:.2f}",f"{batch_time:.3f}",
                     f"{gpu_mem:.1f}",f"{opt.param_groups[0]['lr']:.6f}"])
        csvf.flush()

        print(f"Epoch {epoch:02d} ValF1={val_f1:.4f} ValAcc={val_acc:.4f} "
              f"TrainF1={tr_f1:.4f} LR={opt.param_groups[0]['lr']:.2e}")

        # ----- checkpoints -----
        ckpt_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "best_f1": best_f1,
            "wait": wait,
        }
        torch.save(ckpt_state, ckpt_last)

        if val_f1 > best_f1 + 1e-4:
            best_f1, wait = val_f1, 0
            torch.save(model.state_dict(), ckpt_best)
        else:
            wait += 1
            if wait >= cfg.early_patience:
                print("Early stopped."); break

    csvf.close(); wandb.finish()


# ------------- launcher -------------
if __name__ == "__main__":
    import multiprocessing as mp; mp.freeze_support()
    main(CONFIG)
