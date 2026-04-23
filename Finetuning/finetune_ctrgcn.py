"""
finetune_ctrgcn.py  ── v2 "Boost" Edition
==========================================
Mọi kỹ thuật được áp dụng để CTR-GCN vượt ST-GCN:

  1. Augmentation phong phú hơn:
       - Temporal random crop (thay vì chỉ sliding-window cố định)
       - Spatial rotation (±15°)
       - Scale jitter (±10%)
       - Joint drop-out (mask 1-2 joints ngẫu nhiên)
       - Flip LR + Gaussian noise (như cũ)

  2. WeightedRandomSampler — oversample lớp falling để cân bằng batch

  3. Warmup LR (10 epoch linear) + CosineAnnealing

  4. AdamW với weight_decay=1e-3 (ổn định hơn Adam với mạng sâu)

  5. Boost falling weight × 3.0 trong FocalLoss

  6. Stride = 10 (nhiều windows hơn từ cùng data)

  7. Multi-view TTA (flip + scale 0.95 + scale 1.05) → average 3 views

  8. Patience = 60, epochs = 250

  9. EMA (Exponential Moving Average) model để smooth weight

  10. Gradient clipping = 3.0 (ketat hơn)
"""

from __future__ import annotations
import argparse, pickle, sys, json, random, math, warnings, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from Models.ctrgcn_model import CTRGCN

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(r"d:\Capstone2026\Action Predict\Labeled_data")
DATA_DIR  = BASE_DIR / "stgcn_augmented"
SAVE_DIR  = BASE_DIR / "checkpoints_v4"

CLASS_NAMES = ["standing", "walking", "sitting", "falling"]
FALLING_IDX = 3
FLIP_PAIRS  = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
NUM_JOINTS  = 17


# ─── Augmentation ──────────────────────────────────────────────────────────────
def aug_flip(d):
    out = d.copy(); out[0] = -out[0]
    tmp = out.copy()
    for l, r in FLIP_PAIRS:
        out[:,:,l,:] = tmp[:,:,r,:]; out[:,:,r,:] = tmp[:,:,l,:]
    return out

def aug_noise(d, sigma=0.015):
    out = d.copy(); mask = out[2:3] > 0
    out[0:2] += np.random.randn(2, *out.shape[1:]).astype(np.float32) * sigma * mask
    return out

def aug_rotate(d, max_deg=15.0):
    """Rotate skeleton in XY plane by random angle."""
    out = d.copy()
    angle = np.random.uniform(-max_deg, max_deg) * math.pi / 180.0
    c, s  = math.cos(angle), math.sin(angle)
    x, y  = out[0].copy(), out[1].copy()
    out[0] = c * x - s * y
    out[1] = s * x + c * y
    return out

def aug_scale(d, lo=0.90, hi=1.10):
    """Uniform scale skeleton XY coordinates."""
    out = d.copy()
    scale = np.random.uniform(lo, hi)
    out[0:2] *= scale
    return out

def aug_joint_dropout(d, p=0.10):
    """Zero-out each joint independently with probability p."""
    out = d.copy()
    mask = np.random.rand(NUM_JOINTS) < p     # (V,)
    out[:, :, mask, :] = 0.0
    return out

def aug_temporal_crop(d, clip_len):
    """Random temporal crop — returns exactly clip_len frames."""
    C, T, V, M = d.shape
    if T <= clip_len:
        pad = np.zeros((C, clip_len - T, V, M), dtype=np.float32)
        return np.concatenate([d, pad], axis=1)
    start = np.random.randint(0, T - clip_len + 1)
    return d[:, start:start + clip_len]


# ─── Sliding window ─────────────────────────────────────────────────────────────
def make_windows(data, labels, clip_len, stride):
    N, C, T, V, M = data.shape
    segs, labs = [], []
    for i in range(N):
        for start in range(0, max(1, T - clip_len + 1), stride):
            end = start + clip_len
            seg = data[i, :, start:end] if end <= T else data[i]
            if seg.shape[1] < clip_len:
                pad = np.zeros((C, clip_len - seg.shape[1], V, M), dtype=np.float32)
                seg = np.concatenate([seg, pad], axis=1)
            segs.append(seg); labs.append(labels[i])
    return np.stack(segs), labs


# ─── Dataset ────────────────────────────────────────────────────────────────────
class SkeletonDS(Dataset):
    def __init__(self, data, labels, clip_len, is_train=True):
        self.raw      = data          # keep full temporal dimension for random crop
        self.labels   = labels
        self.clip_len = clip_len
        self.is_train = is_train
        # For val: fixed center crop
        idx = np.linspace(0, data.shape[2]-1, clip_len, dtype=int)
        self.fixed = data[:, :, idx]  # (N, C, clip_len, V, M)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        if self.is_train:
            # Random temporal crop from full sequence
            d = aug_temporal_crop(self.raw[i], self.clip_len)   # (C, T, V, M)
            if random.random() < 0.5: d = aug_flip(d)
            d = aug_noise(d, sigma=random.uniform(0.005, 0.025))
            if random.random() < 0.5: d = aug_rotate(d, max_deg=15.0)
            if random.random() < 0.4: d = aug_scale(d, 0.90, 1.10)
            if random.random() < 0.3: d = aug_joint_dropout(d, p=0.10)
        else:
            d = self.fixed[i].copy()
        d = d.squeeze(-1)             # (C, T, V)
        return torch.from_numpy(d.astype(np.float32)), self.labels[i]


# ─── EMA ────────────────────────────────────────────────────────────────────────
class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.ema   = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()


# ─── Loss ───────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ls=0.15):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.ls = ls

    def forward(self, x, y):
        ce = F.cross_entropy(x, y, weight=self.weight, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-F.cross_entropy(x, y, reduction='none'))
        return (((1-pt)**self.gamma) * ce).mean()

def class_weights(labels, nc, boost_falling=3.0):
    c = Counter(labels); total = len(labels)
    w = torch.zeros(nc)
    for i in range(nc):
        w[i] = total / (nc * max(c.get(i, 1), 1))
    w[FALLING_IDX] *= boost_falling
    return w / w.sum() * nc

def make_sampler(labels):
    """WeightedRandomSampler: oversample under-represented classes."""
    c   = Counter(labels)
    wts = [1.0 / c[l] for l in labels]
    return WeightedRandomSampler(wts, num_samples=len(wts), replacement=True)

def mixup(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam


# ─── Warmup scheduler ───────────────────────────────────────────────────────────
class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.opt          = optimizer
        self.warmup       = warmup_epochs
        self.total        = total_epochs
        self.base_lr      = base_lr
        self.min_lr       = min_lr
        self._epoch       = 0

    def step(self):
        self._epoch += 1
        e = self._epoch
        if e <= self.warmup:
            lr = self.base_lr * e / self.warmup
        else:
            prog = (e - self.warmup) / max(1, self.total - self.warmup)
            lr   = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * prog))
        for g in self.opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)
        return lr


# ─── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(preds, labels, nc):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(labels, preds)
    p, r, f, s = precision_recall_fscore_support(
        labels, preds, labels=list(range(nc)), zero_division=0, average=None)
    per_class = {CLASS_NAMES[i]: {"p": float(p[i]), "r": float(r[i]),
                                   "f1": float(f[i]), "support": int(s[i])}
                 for i in range(nc)}
    return {"accuracy": float(acc), "macro_f1": float(f.mean()),
            "macro_p": float(p.mean()), "macro_r": float(r.mean()),
            "fall_recall": float(r[FALLING_IDX]),
            "fall_f1": float(f[FALLING_IDX]),
            "per_class": per_class}


# ─── Multi-view TTA evaluate ────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, nc):
    model.eval()
    total_loss = total = 0; preds, labs = [], []
    for data, labels in loader:
        data = data.to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.to(device)

        # View 1: original
        v1 = data
        # View 2: flip LR
        v2 = data.clone(); v2[:, 0] = -v2[:, 0]
        for l, r in FLIP_PAIRS:
            tmp = v2[:,:,:,l].clone(); v2[:,:,:,l] = v2[:,:,:,r]; v2[:,:,:,r] = tmp
        # View 3: slight scale up
        v3 = data * 1.05

        logits = (model(v1) + model(v2) + model(v3)) / 3.0
        loss   = criterion(logits, labels)
        total_loss += loss.item() * data.size(0); total += data.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        labs.extend(labels.cpu().tolist())
    return total_loss / max(total, 1), compute_metrics(preds, labs, nc)


# ─── Train one epoch ────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device, ema=None, alpha=0.3, clip_norm=3.0):
    model.train()
    total_loss = total = correct = 0
    for data, labels in loader:
        data = data.to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.to(device)
        data, ya, yb, lam = mixup(data, labels, alpha)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(data)
            loss   = lam * criterion(logits, ya) + (1-lam) * criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        bs = data.size(0); total_loss += loss.item()*bs; total += bs
        correct += (logits.argmax(1) == ya).sum().item()
    return total_loss / max(total, 1), correct / max(total, 1)


# ─── Args ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip-len",      type=int,   default=100)
    p.add_argument("--stride",        type=int,   default=10)
    p.add_argument("--epochs",        type=int,   default=250)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--lr-head",       type=float, default=3e-3)
    p.add_argument("--weight-decay",  type=float, default=1e-3)
    p.add_argument("--mixup-alpha",   type=float, default=0.3)
    p.add_argument("--dropout",       type=float, default=0.3)
    p.add_argument("--patience",      type=int,   default=60)
    p.add_argument("--warmup",        type=int,   default=10)
    p.add_argument("--ema-decay",     type=float, default=0.995)
    p.add_argument("--num-classes",   type=int,   default=4)
    p.add_argument("--no-wandb",      action="store_true")
    p.add_argument("--wandb-project", type=str,   default="ctrgcn-augmented")
    return p.parse_args()


# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # WandB
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"ctrgcn-boost-clip{args.clip_len}-stride{args.stride}",
                config=vars(args),
                tags=["ctrgcn", "boost", "ema", "mstcn", "weighted-sampler"],
            )
            print(f"[WandB] {wandb_run.url}")
        except Exception as e:
            print(f"[WandB] Failed: {e}")

    # Load data
    print("\n[Data] Loading augmented dataset...")
    tr_data = np.load(DATA_DIR / "train_data.npy")
    va_data = np.load(DATA_DIR / "val_data.npy")
    with open(DATA_DIR / "train_label.pkl", "rb") as f: _, tr_labs = pickle.load(f)
    with open(DATA_DIR / "val_label.pkl",   "rb") as f: _, va_labs = pickle.load(f)
    tr_labs = list(tr_labs); va_labs = list(va_labs)

    print(f"  Train: {tr_data.shape}  dist={dict(sorted(Counter(tr_labs).items()))}")
    print(f"  Val  : {va_data.shape}  dist={dict(sorted(Counter(va_labs).items()))}")

    # Sliding window (for val only; train uses random crop inside Dataset)
    tr_win, tr_wlabs = make_windows(tr_data, tr_labs, args.clip_len, args.stride)
    va_win, va_wlabs = make_windows(va_data, va_labs, args.clip_len, args.clip_len)
    print(f"  Train windows: {len(tr_wlabs)}  Val windows: {len(va_wlabs)}")
    print(f"  Train dist: {dict(sorted(Counter(tr_wlabs).items()))}")

    train_ds = SkeletonDS(tr_win, tr_wlabs, args.clip_len, is_train=True)
    val_ds   = SkeletonDS(va_win, va_wlabs, args.clip_len, is_train=False)

    sampler      = make_sampler(tr_wlabs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # Model + EMA
    model = CTRGCN(in_channels=3, num_classes=args.num_classes, dropout=args.dropout).to(device)
    ema   = ModelEMA(model, decay=args.ema_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] CTR-GCN (boost)  |  params: {n_params:,}")

    # Loss & Optimizer (AdamW)
    cw        = class_weights(tr_wlabs, args.num_classes, boost_falling=3.0).to(device)
    criterion = FocalLoss(weight=cw, gamma=2.0, ls=0.15)

    backbone_params = [p for n, p in model.named_parameters() if "fc." not in n]
    head_params     = list(model.fc.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr,      "lr_mult": 1.0},
        {"params": head_params,     "lr": args.lr_head, "lr_mult": args.lr_head / args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup,
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=1e-6,
    )

    # Training loop
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = SAVE_DIR / "best_ctrgcn_augmented.pth"
    best_f1 = 0.0; no_improve = 0

    print(f"\n[Train] {args.epochs} epochs | clip={args.clip_len} | stride={args.stride} | warmup={args.warmup}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, device,
                             ema=ema, alpha=args.mixup_alpha)
        # Evaluate with EMA model
        vl, vm = evaluate(ema.ema, val_loader, criterion, device, args.num_classes)
        lr     = scheduler.step()

        va  = vm["accuracy"]; vf1 = vm["macro_f1"]
        fr  = vm["fall_recall"]; ff1 = vm["fall_f1"]

        print(f"Ep {epoch:3d}/{args.epochs} | "
              f"tl={tl:.4f} ta={ta:.3f} | "
              f"val acc={va:.3f} f1={vf1:.3f} | "
              f"fall R={fr:.3f} F1={ff1:.3f} | lr={lr:.2e}")

        if wandb_run:
            wandb_run.log({
                "train/loss": tl, "train/accuracy": ta,
                "val/loss": vl,   "val/accuracy": va,
                "val/macro_f1": vf1, "val/macro_p": vm["macro_p"],
                "val/macro_r": vm["macro_r"],
                "val/fall_recall": fr, "val/fall_f1": ff1,
                "lr": lr,
            })

        if vf1 > best_f1:
            best_f1 = vf1
            torch.save({"epoch": epoch,
                        "model_state_dict": ema.ema.state_dict(),
                        "metrics": vm}, ckpt_path)
            no_improve = 0
            print(f"  *** New best F1={best_f1:.4f} saved (EMA) ***")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  [EarlyStop] epoch {epoch}")
                break

    # Final report
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    best_metrics = ckpt["metrics"]
    print(f"\n{'='*60}")
    print(f"BEST CTR-GCN (Boost)  (epoch {ckpt['epoch']})")
    print(f"  Accuracy   : {best_metrics['accuracy']:.4f}")
    print(f"  Macro F1   : {best_metrics['macro_f1']:.4f}")
    print(f"  Fall Recall: {best_metrics['fall_recall']:.4f}")
    print(f"  Fall F1    : {best_metrics['fall_f1']:.4f}")
    print("\nPer-class:")
    for cls, m in best_metrics["per_class"].items():
        print(f"  {cls:10s}  P={m['p']:.3f}  R={m['r']:.3f}  F1={m['f1']:.3f}  support={m['support']}")

    json_path = SAVE_DIR / "best_metrics_ctrgcn_augmented.json"
    with open(json_path, "w") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"\n[Saved] {ckpt_path}")
    print(f"[Saved] {json_path}")

    if wandb_run:
        wandb_run.summary["best_accuracy"]    = best_metrics["accuracy"]
        wandb_run.summary["best_macro_f1"]    = best_metrics["macro_f1"]
        wandb_run.summary["best_fall_recall"]  = best_metrics["fall_recall"]
        wandb_run.finish()


if __name__ == "__main__":
    main()
