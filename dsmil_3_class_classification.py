# =========================================================
# DSMIL_GRADES_run.py — FULLY ORDERED SINGLE-FILE SCRIPT
# =========================================================

# -------------------- Imports --------------------
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

from dataset_GRADES import TransATAC_Dataset
from src.models.layers import create_mlp


# -------------------- Device --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =========================================================
# Core DSMIL Components
# =========================================================

class IClassifier(nn.Module):
    """Instance-level classifier (multi-class)."""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, h):
        # h: [B, M, D]
        return self.fc(h)  # [B, M, C]


class BClassifier(nn.Module):
    """Bag-level classifier with top-k per class."""
    def __init__(self, in_dim, attn_dim=384, dropout=0.25, top_k=1):
        super().__init__()
        self.q = nn.Linear(in_dim, attn_dim)
        self.v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim)
        )
        self.norm = nn.LayerNorm(in_dim)
        self.top_k = top_k

    def forward(self, h, instance_logits, attn_mask=None):
        """
        h: [B, M, D]
        instance_logits: [B, M, C]
        """
        B, M, D = h.shape
        C = instance_logits.shape[-1]

        V = self.v(h)          # [B, M, D]
        Q = self.q(h)          # [B, M, A]

        # -------- top-k instance selection per class --------
        top_feats = []
        for b in range(B):
            per_class_feats = []
            for c in range(C):
                scores = instance_logits[b, :, c]   # [M]
                k = min(self.top_k, M)
                idx = torch.topk(scores, k=k).indices
                per_class_feats.append(h[b, idx].mean(dim=0))
            top_feats.append(torch.stack(per_class_feats))
        top_feats = torch.stack(top_feats)  # [B, C, D]

        Q_max = self.q(top_feats)  # [B, C, A]

        # -------- attention --------
        A = torch.bmm(Q, Q_max.transpose(1, 2))  # [B, M, C]
        A = A / torch.sqrt(torch.tensor(Q.shape[-1], device=h.device))

        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(-1) * torch.finfo(A.dtype).min

        A = F.softmax(A, dim=1)

        # -------- aggregation --------
        bag_feats = torch.bmm(A.transpose(1, 2), V)  # [B, C, D]
        bag_feats = self.norm(bag_feats)

        return bag_feats, A


# =========================================================
# DSMIL Model (NOT abstract)
# =========================================================

class DSMIL(nn.Module):
    def __init__(
        self,
        in_dim=1536,
        embed_dim=512,
        num_fc_layers=1,
        dropout=0.25,
        attn_dim=384,
        num_classes=3,
        top_k=1
    ):
        super().__init__()

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            out_dim=embed_dim,
            dropout=dropout,
            end_with_fc=False
        )

        self.i_classifier = IClassifier(embed_dim, num_classes)
        self.b_classifier = BClassifier(
            embed_dim, attn_dim=attn_dim, dropout=dropout, top_k=top_k
        )

        self.classifier = nn.Conv1d(num_classes, num_classes, kernel_size=embed_dim)

    def forward(self, x, label=None, loss_fn=None, return_attention=False):
        """
        x: [B, M, D]
        """
        h = self.patch_embed(x)
        inst_logits = self.i_classifier(h)

        bag_feats, attn = self.b_classifier(h, inst_logits)

        bag_logits = self.classifier(bag_feats).squeeze(-1)  # [B, C]
        max_inst_logits, _ = torch.max(inst_logits, dim=1)

        logits = 0.5 * (bag_logits + max_inst_logits)

        loss = None
        if label is not None and loss_fn is not None:
            loss = loss_fn(logits, label)

        out = {"logits": logits, "loss": loss}
        log = {}
        if return_attention:
            log["attention"] = attn

        return out, log


# =========================================================
# Dataloader utils
# =========================================================

def custom_collate_fn(batch):
    features = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return features, targets


# =========================================================
# Cross-validation function (DEFINED BEFORE USE!)
# =========================================================

def run_cross_validation_dsmil(
    cv_dataset,
    best_params,
    n_folds=5,
    epochs=100,
    resume_from_checkpoint=True
):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(cv_dataset)))):
        print(f"\n===== DSMIL FOLD {fold+1} / {n_folds} =====")

        train_loader = DataLoader(
            Subset(cv_dataset, train_idx),
            batch_size=1,
            shuffle=True,
            collate_fn=custom_collate_fn
        )
        val_loader = DataLoader(
            Subset(cv_dataset, val_idx),
            batch_size=1,
            shuffle=False,
            collate_fn=custom_collate_fn
        )

        model = DSMIL(
            in_dim=best_params["input_dim"],
            embed_dim=best_params["embed_dim"],
            num_fc_layers=best_params["num_fc_layers"],
            dropout=best_params["dropout"],
            attn_dim=best_params["attn_dim"],
            num_classes=3,
            top_k=best_params["top_k"]
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"]
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        loss_fn = nn.CrossEntropyLoss()

        os.makedirs("dsmil_grades_log", exist_ok=True)
        ckpt_path = f"dsmil_grades_log/best_model_fold{fold+1}.pt"

        best_val = float("inf")
        start_epoch = 0
        patience, counter = 7, 0

        if resume_from_checkpoint and os.path.exists(ckpt_path):
            print("Resuming from checkpoint")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            best_val = ckpt["best_val"]
            start_epoch = ckpt["epoch"] + 1

        for epoch in range(start_epoch, epochs):
            model.train()
            train_losses = []

            for feats_list, tgt_list in tqdm(train_loader, leave=False):
                feats = torch.stack(
                    [torch.tensor(f, dtype=torch.float32) for f in feats_list]
                ).to(DEVICE)
                labels = torch.tensor(
                    [t["tumor grade"] for t in tgt_list],
                    dtype=torch.long
                ).to(DEVICE)

                optimizer.zero_grad()
                out, _ = model(feats, label=labels, loss_fn=loss_fn)
                loss = out["loss"]
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            with torch.no_grad():
                for feats_list, tgt_list in val_loader:
                    feats = torch.stack(
                        [torch.tensor(f, dtype=torch.float32) for f in feats_list]
                    ).to(DEVICE)
                    labels = torch.tensor(
                        [t["tumor grade"] for t in tgt_list],
                        dtype=torch.long
                    ).to(DEVICE)

                    out, _ = model(feats, label=labels, loss_fn=loss_fn)
                    val_losses.append(out["loss"].item())

            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train {np.mean(train_losses):.4f} | "
                f"Val {val_loss:.4f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val": best_val
                }, ckpt_path)
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping")
                    break

        fold_losses.append(best_val)

    print("\nFINAL DSMIL RESULTS")
    print("Fold losses:", fold_losses)
    print("Mean loss:", np.mean(fold_losses))
    return fold_losses


# =========================================================
# MAIN (ALWAYS LAST)
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

    cv_dataset = TransATAC_Dataset(
        train_df,
        feature_dir,
        score_columns=["tumor grade"]
    )

    best_params = {
        "input_dim": 1536,
        "embed_dim": 512,
        "num_fc_layers": 1,
        "dropout": 0.25,
        "attn_dim": 384,
        "lr": 1e-5,
        "weight_decay": 1e-5,
        "top_k": 1
    }

    results = run_cross_validation_dsmil(
        cv_dataset,
        best_params,
        n_folds=5,
        epochs=100,
        resume_from_checkpoint=True
    )
