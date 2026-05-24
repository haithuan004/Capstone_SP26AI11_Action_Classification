"""
finetune_v3.py
==============
Finetune ST-GCN trên dataset V5 (Optimal Clamp-Padding).
- Data source : stgcn_augmented_v5/
- Pretrained  : Raw backbone (ntu60-xsub) thay vì overfit weights cũ
- Loss        : FocalLoss + Class Weights dựa trên Window counts
"""

from __future__ import annotations
import argparse, pickle, sys, json, random, math, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "Models"))
from stgcn_model import STGCN, load_pretrained_backbone

# ─── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"d:\Capstone2026\Action Predict\Labeled_data")
DATA_DIR   = BASE_DIR / "stgcn_raw_data"
PRETRAINED = BASE_DIR / "stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221129-c4b44488.pth"
SAVE_DIR   = BASE_DIR / "checkpoints_v6_raw"

CLASS_NAMES  = ["standing", "walking", "sitting", "falling"]
FALLING_IDX  = 3
FLIP_PAIRS   = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]

# Online Augmentation (Tăng cường động trong quá trình Train)
def aug_flip(data):
    out = data.copy(); out[0] = -out[0]
    tmp = out.copy()
    for l, r in FLIP_PAIRS:
        out[:,:,l,:] = tmp[:,:,r,:]; out[:,:,r,:] = tmp[:,:,l,:]
    return out

def aug_noise(data, sigma=0.01):
    out = data.copy(); mask = out[2:3] > 0
    out[0:2] += np.random.randn(2, *out.shape[1:]).astype(np.float32) * sigma * mask
    return out

# ─── Sliding window ─────────────────────────────────────────────────────────────
def make_windows(data, labels, clip_len, stride):
    N, C, T, V, M = data.shape
    segs, labs = [], []
    for i in range(N):
        # Sliding window over the clamp-padded sequence
        for start in range(0, max(1, T - clip_len + 1), stride):
            end = start + clip_len
            seg = data[i, :, start:end] if end <= T else data[i]
            
            # Fallback if somehow still shorter than clip_len (e.g. T < clip_len)
            if seg.shape[1] < clip_len:
                diff = clip_len - seg.shape[1]
                last_frame = seg[:, -1:, :, :]
                pad = np.repeat(last_frame, diff, axis=1)
                seg = np.concatenate([seg, pad], axis=1)
                
            segs.append(seg)
            labs.append(labels[i])
    return np.stack(segs), labs

# ─── Dataset ────────────────────────────────────────────────────────────────────
class SkeletonDS(Dataset):
    def __init__(self, data, labels, is_train=False):
        self.data = data
        self.labels = labels
        self.is_train = is_train
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        # Online Augmentation (Chỉ áp dụng khi Train)
        if self.is_train:
            if random.random() > 0.5:
                x = aug_flip(x)
            if random.random() > 0.5:
                x = aug_noise(x, sigma=0.015)
        x = x.squeeze(-1)        
        return torch.from_numpy(x).float(), y

# ─── Loss ───────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ls=0.2):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.ls = ls

    def forward(self, x, y):
        ce = F.cross_entropy(x, y, weight=self.weight, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-F.cross_entropy(x, y, reduction='none'))
        return (((1-pt)**self.gamma)*ce).mean()

def class_weights(labels, nc, boost_falling=1.5):
    c = Counter(labels); total = len(labels)
    w = torch.zeros(nc)
    for i in range(nc):
        w[i] = total / (nc * max(c.get(i,1), 1))
    w[FALLING_IDX] *= boost_falling
    return w / w.sum() * nc

def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam*x + (1-lam)*x[idx], y, y[idx], lam

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

# ─── Evaluate with TTA ──────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, nc):
    model.eval()
    total_loss = total = 0; preds, labs = [], []
    for data, labels in loader:
        data = data.to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.to(device)
        # TTA flip
        df = data.clone(); df[:, 0] = -df[:, 0]
        for l, r in FLIP_PAIRS:
            tmp = df[:,:,:,l].clone(); df[:,:,:,l] = df[:,:,:,r]; df[:,:,:,r] = tmp
        logits = (model(data) + model(df)) / 2.0
        loss   = criterion(logits, labels)
        total_loss += loss.item()*data.size(0); total += data.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        labs.extend(labels.cpu().tolist())
    return total_loss/max(total,1), compute_metrics(preds, labs, nc)

# ─── Train one epoch ────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device, alpha=0.2, clip_norm=5.0):
    model.train()
    total_loss = total = correct = 0
    for data, labels in loader:
        data = data.to(device)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.to(device)
        data, ya, yb, lam = mixup(data, labels, alpha)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            logits = model(data)
            loss   = lam*criterion(logits, ya) + (1-lam)*criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        bs = data.size(0); total_loss += loss.item()*bs; total += bs
        correct += (logits.argmax(1) == ya).sum().item()
    return total_loss/max(total,1), correct/max(total,1)

# ─── Main ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clip-len",     type=int,   default=100)
    p.add_argument("--stride",       type=int,   default=10) # Denser stride for more data
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=5e-5) # Cho 2 block cuối
    p.add_argument("--lr-head",      type=float, default=5e-4) # Cho Head
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--mixup-alpha",  type=float, default=0.4) # Tăng mixup chống overfit
    p.add_argument("--dropout",      type=float, default=0.6) # Tăng dropout chống overfit
    p.add_argument("--patience",     type=int,   default=40)
    p.add_argument("--num-classes",  type=int,   default=4)
    p.add_argument("--no-wandb",     action="store_true")
    p.add_argument("--wandb-project",type=str,   default="stgcn-augmented")
    return p.parse_args()

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
                name=f"stgcn-v9-raw",
                config=vars(args),
                tags=["stgcn", "raw", "weighted_sampler"],
            )
            print(f"[WandB] {wandb_run.url}")
        except Exception as e:
            print(f"[WandB] Failed: {e}")

    # Load data
    print("\n[Data] Loading augmented dataset V5...")
    tr_data = np.load(DATA_DIR / "train_data.npy")
    va_data = np.load(DATA_DIR / "val_data.npy")
    with open(DATA_DIR / "train_label.pkl", "rb") as f: _, tr_labs = pickle.load(f)
    with open(DATA_DIR / "val_label.pkl",   "rb") as f: _, va_labs = pickle.load(f)
    tr_labs = list(tr_labs); va_labs = list(va_labs)

    print(f"  Train: {tr_data.shape}  dist={dict(sorted(Counter(tr_labs).items()))}")
    print(f"  Val  : {va_data.shape}  dist={dict(sorted(Counter(va_labs).items()))}")

    # Sliding window
    tr_wdata, tr_wlabs = make_windows(tr_data, tr_labs, args.clip_len, args.stride)
    vl_wdata, vl_wlabs = make_windows(va_data, va_labs, args.clip_len, stride=50) 
    
    # ─── dataset & loader ─────────────────────────────────────────────────────
    print(f"[Data] Loading Raw dataset V6...")
    print(f"  Train windows: {len(tr_wdata)}  Val windows: {len(vl_wdata)}")
    
    # Tính trọng số cho WeightedRandomSampler
    class_counts = Counter(tr_wlabs)
    print(f"  Train window dist: {dict(class_counts)}")
    
    total_samples = len(tr_wlabs)
    class_weights_dict = {c: total_samples / count for c, count in class_counts.items()}
    sample_weights = [class_weights_dict[int(lbl)] for lbl in tr_wlabs]
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=True
    )

    train_ds = SkeletonDS(tr_wdata, tr_wlabs, is_train=True)
    val_ds   = SkeletonDS(vl_wdata, vl_wlabs, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = STGCN(in_channels=3, num_classes=args.num_classes, dropout=args.dropout).to(device)
    if PRETRAINED.exists():
        print(f"[Pretrained] Loading general backbone from {PRETRAINED}")
        load_pretrained_backbone(model, str(PRETRAINED), verbose=True)
    else:
        print("[Pretrained] Not found, training from scratch.")

    # Freeze toàn bộ Backbone
    print("[Transfer Learning] Phase 2: Freezing early layers...")
    for param in model.parameters():
        param.requires_grad = False
        
    # Mở khóa 2 block GCN cuối cùng
    print("[Transfer Learning] Unfreezing the last 2 GCN blocks...")
    for param in model.gcn[-2:].parameters():
        param.requires_grad = True
        
    # Mở khóa Head
    print("[Transfer Learning] Unfreezing the Classification Head...")
    for param in model.fc.parameters():
        param.requires_grad = True

    # Loss & Optimizer
    cw        = class_weights(tr_wlabs, args.num_classes).to(device)
    criterion = FocalLoss(weight=cw, gamma=2.0, ls=0.2)
    
    # Optimizer nhận 2 mức LR khác nhau
    optimizer = torch.optim.AdamW([
        {"params": model.gcn[-2:].parameters(), "lr": args.lr},
        {"params": model.fc.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training loop
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = SAVE_DIR / "best_stgcn_v9_raw.pth"
    best_f1 = 0.0; no_improve = 0

    print(f"\n[Train] {args.epochs} epochs | clip={args.clip_len} | stride={args.stride}")
    print("-" * 72)

    for epoch in range(1, args.epochs + 1):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, device, args.mixup_alpha)
        vl, vm = evaluate(model, val_loader, criterion, device, args.num_classes)
        scheduler.step()

        va  = vm["accuracy"]; vf1 = vm["macro_f1"]
        fr  = vm["fall_recall"]; ff1 = vm["fall_f1"]
        lr  = optimizer.param_groups[0]["lr"]

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
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "metrics": vm}, ckpt_path)
            no_improve = 0
            print(f"  *** New best F1={best_f1:.4f} saved ***")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  [EarlyStop] epoch {epoch}")
                break

    # Final report
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    best_metrics = ckpt["metrics"]
    print(f"\n{'='*60}")
    print(f"BEST MODEL  (epoch {ckpt['epoch']})")
    print(f"  Accuracy   : {best_metrics['accuracy']:.4f}")
    print(f"  Macro F1   : {best_metrics['macro_f1']:.4f}")
    print(f"  Fall Recall: {best_metrics['fall_recall']:.4f}")
    print(f"  Fall F1    : {best_metrics['fall_f1']:.4f}")
    print(f"\nPer-class:")
    for cls, m in best_metrics["per_class"].items():
        print(f"  {cls:10s}  P={m['p']:.3f}  R={m['r']:.3f}  F1={m['f1']:.3f}  support={m['support']}")

    # Save JSON
    json_path = SAVE_DIR / "best_metrics_v9_raw.json"
    with open(json_path, "w") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"\n[Saved] {ckpt_path}")
    print(f"[Saved] {json_path}")

    if wandb_run:
        wandb_run.summary["best_accuracy"]   = best_metrics["accuracy"]
        wandb_run.summary["best_macro_f1"]   = best_metrics["macro_f1"]
        wandb_run.summary["best_fall_recall"] = best_metrics["fall_recall"]
        wandb_run.finish()


if __name__ == "__main__":
    main()
