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
from scripts.diagset_data import DiagSetAPatchDataset

from scripts.label_schemes import (
    collapse_to_S1,
    collapse_to_S2,
    collapse_to_S3,
    collapse_to_S4,
)



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


def evaluate_split(cfg: CONFIG, split_name: str = "test", ckpt_path: Optional[str] = None):
    """
    Avalia um modelo treinado em um split específico (por padrão, 'test').

    Usa:
      - mesmas configs de schema/mag/model definidas em cfg
      - ckpt_best_{model}_{schema}.pt por padrão, ou um caminho passado em ckpt_path

    Se cfg.schema == "s5":
      - calcula métricas em S5 (9 classes)
      - deriva e calcula métricas em S1, S2, S3 e S4 a partir dos rótulos S5.
    """
    import pandas as pd
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    from torch.utils.data import DataLoader

    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- dataset ---
    split_dir = Path(f"data/splits_{cfg.split}_{cfg.mag}")
    csv_path  = split_dir / f"{split_name}.csv"
    print(f"[EVAL] Lendo CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    ds = DiagSetAPatchDataset(
        df,
        train=False,                      # sem augment aleatório
        mode=cfg.schema,
        crop_size=cfg.img_size,
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_gpu * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # --- modelo ---
    n_cls = 9 if cfg.schema == "s5" else 2
    model = get_model(cfg.model, n_cls).to(device)

    # escolher checkpoint: best por padrão
    if ckpt_path is None:
        ckpt_dir  = Path("checkpoints") / cfg.mag
        ckpt_file = ckpt_dir / f"ckpt_best_{cfg.model}_{cfg.schema}.pt"
    else:
        ckpt_file = Path(ckpt_path)

    print(f"[EVAL] Carregando checkpoint: {ckpt_file}")
    state = torch.load(ckpt_file, map_location=device)

    # ckpt_best salva só state_dict; ckpt_last salva um dict maior
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_gts   = []
    loss_total = 0.0

    from tqdm import tqdm
    autocast_kwargs = {"device_type": "cuda"} if device == "cuda" else {}

    with torch.no_grad(), torch.autocast(enabled=(device == "cuda"), **autocast_kwargs):
        for b in tqdm(dl, desc=f"EVAL {split_name}"):
            x = b["image"].to(device)
            y = b["label"].to(device)
            out = model(x)
            loss_total += criterion(out, y).item()
            preds = out.argmax(1)
            all_preds.append(preds.cpu())
            all_gts.append(y.cpu())

    # empilha tudo
    all_preds = torch.cat(all_preds).numpy()
    all_gts   = torch.cat(all_gts).numpy()

    loss = loss_total / len(dl)

    # ----------------------------
    # Métricas no esquema nativo
    # ----------------------------
    results = {}

    cm_native = confusion_matrix(all_gts, all_preds, labels=list(range(n_cls)))
    acc_native = accuracy_score(all_gts, all_preds)
    f1_native  = f1_score(all_gts, all_preds, average="macro")

    print(f"[EVAL {split_name}] ({cfg.schema}) loss={loss:.4f} acc={acc_native:.4f} f1={f1_native:.4f}")
    print("Matriz de confusão (schema nativo):")
    print(cm_native)

    results[cfg.schema.upper()] = {
        "loss": float(loss),
        "acc": float(acc_native),
        "f1":  float(f1_native),
        "cm":  cm_native,
    }

    # -----------------------------------------
    # Se for S5, derivar S1–S4 a partir de S5
    # -----------------------------------------
    if cfg.schema == "s5":
        # Cada tupla: (nome_do_schema, função_de_colapso, número_de_classes)
        derived_schemes = [
            ("S1", collapse_to_S1, 2),
            ("S2", collapse_to_S2, 2),
            ("S3", collapse_to_S3, 4),
            ("S4", collapse_to_S4, 6),
        ]

        for name, collapse_fn, n_out in derived_schemes:
            y_true = collapse_fn(all_gts)
            y_pred = collapse_fn(all_preds)

            cm = confusion_matrix(y_true, y_pred, labels=list(range(n_out)))
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="macro")

            print(f"[EVAL {split_name}] (derivado {name}) acc={acc:.4f} f1={f1:.4f}")
            print(f"Matriz de confusão derivada ({name}):")
            print(cm)

            results[name] = {
                "acc": float(acc),
                "f1":  float(f1),
                "cm":  cm,
            }

    return results


# ------------- launcher -------------
if __name__ == "__main__":
    import multiprocessing as mp; mp.freeze_support()
    main(CONFIG)
