# =========================================================
# DSMIL_GRADES_ORDINAL_SOFTATTN_GATED_FOCAL_POSW_THRESH.py
# =========================================================

# -------------------- FORCE GPU 1 --------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# -------------------- Imports --------------------
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset  # <- removed WeightedRandomSampler per your request set
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dataset_GRADES import TransATAC_Dataset

# -------------------- Device --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# =========================================================
# Utils
# =========================================================
def custom_collate_fn(batch):
    feats = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return feats, targets

def compute_pos_weight_per_threshold(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    For ordinal cumulative link with K classes, we have K-1 thresholds.
    For threshold k: positive iff y > k.
    pos_weight = N_neg / N_pos (PyTorch BCEWithLogitsLoss convention).
    Returns shape [K-1].
    """
    K = num_classes
    pos_weights = []
    N = len(labels)
    for k in range(K - 1):
        pos = np.sum(labels > k)
        neg = N - pos
        pos = max(pos, 1)  # avoid div by 0
        pos_weights.append(neg / pos)
    return torch.tensor(pos_weights, dtype=torch.float32)

@torch.no_grad()
def ordinal_logits_to_class_with_thresholds(logits: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
    """
    logits: [B, K-1] where K=3 => 2 thresholds
    probs = sigmoid(logits)
    pred = I(p0 > t0) + I(p1 > t1)
    """
    probs = torch.sigmoid(logits)
    p0 = probs[:, 0]
    p1 = probs[:, 1]
    return (p0 > t0).long() + (p1 > t1).long()

def tune_thresholds_on_val(y_true: np.ndarray, probs_01: np.ndarray):
    """
    Tune thresholds (t0,t1) on validation fold.
    y_true: shape [N] in {0,1,2}
    probs_01: shape [N,2] probabilities for thresholds (y>0, y>1)

    Objective: 0.5*(precision0 + precision2), with monotonic constraint t0 <= t1.
    """
    best = {
        "score": -1.0,
        "t0": 0.5,
        "t1": 0.5,
        "prec0": 0.0,
        "prec2": 0.0
    }

    # reasonably fine grid; adjust if you want faster/slower
    grid = np.linspace(0.15, 0.85, 15)

    p0 = probs_01[:, 0]
    p1 = probs_01[:, 1]

    for t0 in grid:
        for t1 in grid:
            if t0 > t1:
                continue

            y_pred = (p0 > t0).astype(int) + (p1 > t1).astype(int)

            # precision for class 0 and 2; handle zero-division safely
            prec0 = precision_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0]
            prec2 = precision_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
            score = 0.5 * (prec0 + prec2)

            if score > best["score"]:
                best.update(score=score, t0=float(t0), t1=float(t1), prec0=float(prec0), prec2=float(prec2))

    return best

# =========================================================
# ORDINAL FOCAL BCE WITH POS_WEIGHT + MARGIN REGULARIZER
# =========================================================
class OrdinalFocalCumulativeLoss(nn.Module):
    """
    K-class ordinal loss implemented as K-1 binary thresholds.
    Adds:
      - pos_weight per threshold (cost-sensitive)
      - focal modulation (gamma)
      - margin regularizer to discourage ambiguous extreme predictions
        (push p0 away from 0.5, and p1 away from 0.5; and/or discourage high p1 too easily)

    logits: [B, K-1]
    targets: [B] in {0..K-1}
    """
    def __init__(
        self,
        num_classes: int,
        pos_weight: torch.Tensor,
        gamma: float = 2.0,
        margin: float = 0.15,
        lambda_margin: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.margin = margin
        self.lambda_margin = lambda_margin

        # store pos_weight as buffer so it moves with .to(device)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B = logits.size(0)
        K = self.num_classes

        # Build cumulative targets: y>k
        ordinal_targets = torch.zeros(B, K - 1, device=logits.device)
        for k in range(K - 1):
            ordinal_targets[:, k] = (targets > k).float()

        # BCE per threshold with pos_weight
        # reduction='none' to apply focal term
        bce = F.binary_cross_entropy_with_logits(
            logits,
            ordinal_targets,
            pos_weight=self.pos_weight,  # shape [K-1] broadcasts over batch
            reduction="none"
        )  # [B, K-1]

        # Focal modulation
        # pt = p if y=1 else (1-p)
        p = torch.sigmoid(logits)
        pt = p * ordinal_targets + (1 - p) * (1 - ordinal_targets)
        focal = (1 - pt).pow(self.gamma)

        focal_bce = (focal * bce).mean()

        # -------- Margin regularizer (discourage ambiguous extremes) --------
        # Encourage p to be away from 0.5 by at least margin:
        # penalty = relu(margin - |p - 0.5|)
        # This pushes both thresholds toward confident decisions.
        with torch.no_grad():
            pass
        amb_pen = F.relu(self.margin - torch.abs(p - 0.5))  # [B,K-1]
        margin_loss = amb_pen.mean()

        return focal_bce + self.lambda_margin * margin_loss

# =========================================================
# PATCH ENCODER (DEEP)
# =========================================================
class PatchEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# GATED ATTENTION POOLING (tanh ⊙ sigmoid)
# =========================================================
class GatedAttention(nn.Module):
    """
    Standard gated attention used in MIL:
      a = tanh(Vh)
      b = sigmoid(Uh)
      scores = w^T (a ⊙ b)
      weights = softmax(scores / temperature)

    Returns bag embedding and attention weights.
    """
    def __init__(self, dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.V = nn.Linear(dim, dim // 2)
        self.U = nn.Linear(dim, dim // 2)
        self.w = nn.Linear(dim // 2, 1)

    def forward(self, h):
        # h: [B, M, D]
        a = torch.tanh(self.V(h))         # [B, M, D/2]
        b = torch.sigmoid(self.U(h))      # [B, M, D/2]
        g = a * b                         # [B, M, D/2]
        scores = self.w(g).squeeze(-1)    # [B, M]
        scores = scores / self.temperature
        weights = F.softmax(scores, dim=1)           # [B, M]
        bag = torch.bmm(weights.unsqueeze(1), h)     # [B, 1, D]
        return bag.squeeze(1), weights

# =========================================================
# DSMIL (ORDINAL, GATED ATTENTION)
# =========================================================
class DSMIL_Ordinal(nn.Module):
    def __init__(self, in_dim, embed_dim, num_classes, dropout, temperature):
        super().__init__()
        self.encoder = PatchEncoder(in_dim, embed_dim, dropout)
        self.attention = GatedAttention(embed_dim, temperature)
        self.classifier = nn.Linear(embed_dim, num_classes - 1)  # K-1 logits

    def forward(self, x):
        h = self.encoder(x)             # [B, M, D]
        bag, attn = self.attention(h)   # [B, D], [B, M]
        logits = self.classifier(bag)   # [B, K-1]
        return logits, attn

# =========================================================
# CROSS-VALIDATION
# =========================================================
def run_cv(
    dataset,
    labels_all,
    results_root,
    n_folds=5,
    epochs=100
):
    os.makedirs(results_root, exist_ok=True)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(labels_all)):
        fold_id = fold + 1
        print(f"\n===== FOLD {fold_id}/{n_folds} =====")

        fold_dir = Path(results_root) / f"fold_{fold_id}"
        model_dir = fold_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        train_labels = labels_all[train_idx].astype(int)

        # ---------------- DataLoaders ----------------
        # Note: sampler removed. Use cost-sensitive loss instead.
        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=1,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=4
        )

        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=1,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=4
        )

        # ---------------- Model ----------------
        model = DSMIL_Ordinal(
            in_dim=1536,
            embed_dim=512,
            num_classes=3,
            dropout=0.25,
            temperature=1.0  # sharper than 1.5 to reduce spurious diffusion
        ).to(DEVICE)

        # ---------------- Loss: pos_weight per threshold + focal + margin ----------------
        pos_weight = compute_pos_weight_per_threshold(train_labels, num_classes=3).to(DEVICE)
        # For your global counts (0:439, 1:210, 2:146), typical per-threshold pos_weight ~:
        # k=0 positives (1/2) = 356, neg=439 => 1.23
        # k=1 positives (2)   = 146, neg=649 => 4.44
        # But we compute per-fold from train split.

        criterion = OrdinalFocalCumulativeLoss(
            num_classes=3,
            pos_weight=pos_weight,
            gamma=2.0,
            margin=0.15,
            lambda_margin=0.2
        )

        # ---------------- Optimizer + schedule (higher LR) ----------------
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=8
        )

        # We select by 0.5*(precision0 + precision2) with tuned thresholds on val.
        best_sel = -1.0
        best_ba = 0.0
        best_thresholds = {"t0": 0.5, "t1": 0.5, "prec0": 0.0, "prec2": 0.0, "score": 0.0}

        for epoch in range(epochs):
            # ---------------- TRAIN ----------------
            model.train()
            train_losses = []

            for feats_list, tgt_list in tqdm(train_loader, leave=False):
                feats = torch.stack([f.float() for f in feats_list]).to(DEVICE)  # [1, M, D]
                labels = torch.tensor([t["tumor grade"] for t in tgt_list], device=DEVICE).long()

                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(feats)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                train_losses.append(loss.item())

            # ---------------- VALIDATION (get probs, tune thresholds) ----------------
            model.eval()
            y_true, probs01 = [], []

            with torch.no_grad():
                for feats_list, tgt_list in val_loader:
                    feats = torch.stack([f.float() for f in feats_list]).to(DEVICE)
                    labels = torch.tensor([t["tumor grade"] for t in tgt_list], device=DEVICE).long()

                    logits, _ = model(feats)                     # [1,2]
                    p = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [2]
                    probs01.append(p)
                    y_true.append(int(labels.item()))

            y_true = np.array(y_true, dtype=int)
            probs01 = np.vstack(probs01)  # [N,2]

            tuned = tune_thresholds_on_val(y_true, probs01)
            t0, t1 = tuned["t0"], tuned["t1"]

            y_pred = (probs01[:, 0] > t0).astype(int) + (probs01[:, 1] > t1).astype(int)

            ba = balanced_accuracy_score(y_true, y_pred)
            sel = tuned["score"]  # 0.5*(prec0+prec2)

            scheduler.step(sel)

            print(
                f"Epoch {epoch+1:03d} | "
                f"Train {np.mean(train_losses):.4f} | "
                f"Sel 0.5*(P0+P2) {sel:.4f} (P0 {tuned['prec0']:.4f}, P2 {tuned['prec2']:.4f}) | "
                f"BA {ba:.4f} | "
                f"t0 {t0:.2f} t1 {t1:.2f}"
            )

            if sel > best_sel:
                best_sel = sel
                best_ba = ba
                best_thresholds = tuned

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "t0": t0,
                        "t1": t1,
                        "best_sel": best_sel,
                        "best_ba": best_ba,
                        "pos_weight": pos_weight.detach().cpu().numpy().tolist(),
                    },
                    model_dir / "best_model.pt"
                )

        # ---------------- FINAL EVAL WITH BEST CHECKPOINT ----------------
        ckpt = torch.load(model_dir / "best_model.pt", map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        t0, t1 = ckpt["t0"], ckpt["t1"]

        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for feats_list, tgt_list in val_loader:
                feats = torch.stack([f.float() for f in feats_list]).to(DEVICE)
                labels = torch.tensor([t["tumor grade"] for t in tgt_list], device=DEVICE).long()
                logits, _ = model(feats)

                pred = ordinal_logits_to_class_with_thresholds(logits, t0=t0, t1=t1)
                y_true.append(int(labels.item()))
                y_pred.append(int(pred.item()))

        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        df_report = pd.DataFrame(report).transpose()
        df_report["balanced_accuracy"] = best_ba
        df_report["selection_score_0.5_P0_P2"] = best_sel
        df_report["best_t0"] = t0
        df_report["best_t1"] = t1
        df_report.to_csv(fold_dir / "results.csv")

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Fold {fold_id} Confusion Matrix (t0={t0:.2f}, t1={t1:.2f})")
        plt.tight_layout()
        plt.savefig(fold_dir / "results.png")
        plt.close()

        print(
            f"[Fold {fold_id}] Best sel={best_sel:.4f} BA={best_ba:.4f} "
            f"(t0={t0:.2f}, t1={t1:.2f}) pos_weight={ckpt['pos_weight']}"
        )

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    metadata_path = Path(
        ""
    )
    feature_dir = Path(
        ""
    )

    df = pd.read_csv(metadata_path)
    train_df = df[df["train"] == "train"].reset_index(drop=True)

    # IMPORTANT: assumes you've already relabeled grades to {0,1,2} in the CSV.
    labels_all = train_df["tumor grade"].values.astype(int)

    dataset = TransATAC_Dataset(
        train_df,
        feature_dir,
        score_columns=["tumor grade"]
    )

    run_cv(
        dataset=dataset,
        labels_all=labels_all,
        results_root="DSMIL_ORDINAL_GATED_FOCAL_POSW_THRESH_RESULTS",
        n_folds=5,
        epochs=100
    )
