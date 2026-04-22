
"""
=========================================================

Multimodal Classification with Contrastive Learning

Inputs:
    - Image feature tensors
    - Tabular features

Target:
    - Binary label

Features:
    - 5-fold cross validation
    - Optuna hyperparameter search
    - Weights & Biases logging (optional)
    - Contrastive alignment loss
    - Checkpoint saving


=========================================================
"""

# =========================================================
# IMPORTS
# =========================================================
import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import optuna

try:
    import wandb
    USE_WANDB = True
except:
    USE_WANDB = False


# =========================================================
# GLOBALS
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================================================
# ARGPARSE
# =========================================================
parser = argparse.ArgumentParser()

parser.add_argument("--metadata_csv", type=str, default="./data/metadata.csv")
parser.add_argument("--feature_dir", type=str, default="./data/features")
parser.add_argument("--target_col", type=str, default="target")

parser.add_argument(
    "--feature_cols",
    nargs="+",
    default=["feature_1", "feature_2", "feature_3", "feature_4"]
)

parser.add_argument("--n_trials", type=int, default=20)
parser.add_argument("--n_folds", type=int, default=5)
parser.add_argument("--epochs", type=int, default=30)

parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
parser.add_argument("--project_name", type=str, default="multimodal_public")

ARGS = parser.parse_args()

CHECKPOINT_DIR = Path(ARGS.checkpoint_dir)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# DATASET
# =========================================================
class DatasetClass(Dataset):

    def __init__(self, df, feature_dir, target_col, feature_cols):

        self.df = df.reset_index(drop=True)
        self.feature_dir = Path(feature_dir)

        self.target_col = target_col
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        sample_id = row["sample_id"]

        feature_path = self.feature_dir / f"{sample_id}.pt"

        image_tensor = torch.load(feature_path)

        target = int(float(row[self.target_col]))

        tabular = np.array(
            [float(row[c]) for c in self.feature_cols],
            dtype=np.float32
        )

        return image_tensor, target, tabular


# =========================================================
# COLLATE
# =========================================================
def collate_fn(batch):

    images = [b[0] for b in batch]
    labels = [b[1] for b in batch]

    tabular = torch.tensor(
        np.stack([b[2] for b in batch]),
        dtype=torch.float32
    )

    return images, labels, tabular


def stack_features(feats):

    return torch.stack([
        x.float() if torch.is_tensor(x)
        else torch.tensor(x).float()
        for x in feats
    ])


# =========================================================
# STRATIFIED FOLDS
# =========================================================
def make_folds(df, target_col, n_folds):

    y = df[target_col].astype(int).values

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    np.random.shuffle(idx0)
    np.random.shuffle(idx1)

    folds = []

    for i in range(n_folds):

        val_idx = np.concatenate([
            idx0[i::n_folds],
            idx1[i::n_folds]
        ])

        tr_idx = np.setdiff1d(np.arange(len(df)), val_idx)

        folds.append((tr_idx, val_idx))

    return folds


# =========================================================
# SMALL MLP
# =========================================================
def create_mlp(in_dim, hidden, out_dim, dropout):

    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),

        nn.Linear(hidden, out_dim)
    )


# =========================================================
# IMAGE ENCODER
# =========================================================
class ImageEncoder(nn.Module):

    def __init__(self, params):
        super().__init__()

        self.fc = create_mlp(
            params["input_dim"],
            params["embed_dim"],
            params["embed_dim"],
            params["dropout"]
        )

    def forward(self, x):

        # x: B x P x D
        x = self.fc(x)

        x = x.mean(dim=1)

        return x


# =========================================================
# MODEL
# =========================================================
class MultiModalModel(nn.Module):

    def __init__(self, params, n_features):
        super().__init__()

        emb = params["embed_dim"]

        self.image_encoder = ImageEncoder(params)

        self.tabular_encoder = create_mlp(
            n_features,
            params["tab_hidden"],
            emb,
            params["dropout"]
        )

        self.classifier = nn.Sequential(
            nn.Linear(emb, emb // 2),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(emb // 2, 2)
        )

        self.temp = params["temperature"]

    def contrastive_loss(self, z1, z2):

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        logits = z1 @ z2.T / self.temp

        labels = torch.arange(len(z1)).to(z1.device)

        loss1 = F.cross_entropy(logits, labels)
        loss2 = F.cross_entropy(logits.T, labels)

        return 0.5 * (loss1 + loss2)

    def forward(self, image_feats, tabular, labels=None):

        z_img = self.image_encoder(image_feats)
        z_tab = self.tabular_encoder(tabular)

        c_loss = self.contrastive_loss(z_img, z_tab)

        z = 0.5 * (z_img + z_tab)

        logits = self.classifier(z)

        return logits, c_loss


# =========================================================
# EVALUATE
# =========================================================
@torch.no_grad()
def evaluate(model, loader, lambda_c):

    model.eval()

    losses = []
    preds = []
    probs = []
    labels_all = []

    ce = nn.CrossEntropyLoss()

    for feats, labels, tabular in loader:

        feats = stack_features(feats).to(DEVICE)
        tabular = tabular.to(DEVICE)
        labels = torch.tensor(labels).to(DEVICE)

        logits, c_loss = model(feats, tabular)

        cls = ce(logits, labels)

        loss = cls + lambda_c * c_loss

        losses.append(loss.item())

        p = torch.softmax(logits, dim=1)

        preds.extend(torch.argmax(p, dim=1).cpu().numpy())
        probs.extend(p[:, 1].cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    preds = np.array(preds)
    probs = np.array(probs)
    labels_all = np.array(labels_all)

    bal_acc = balanced_accuracy_score(labels_all, preds)

    try:
        auc = roc_auc_score(labels_all, probs)
    except:
        auc = np.nan

    return np.mean(losses), bal_acc, auc


# =========================================================
# TRAIN ONE FOLD
# =========================================================
def train_fold(trial, fold_id, dataset, tr_idx, va_idx, params):

    train_loader = DataLoader(
        Subset(dataset, tr_idx),
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        Subset(dataset, va_idx),
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    model = MultiModalModel(
        params,
        len(ARGS.feature_cols)
    ).to(DEVICE)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"]
    )

    ce = nn.CrossEntropyLoss()

    best_score = -1

    for epoch in range(ARGS.epochs):

        model.train()

        for feats, labels, tabular in train_loader:

            feats = stack_features(feats).to(DEVICE)
            tabular = tabular.to(DEVICE)
            labels = torch.tensor(labels).to(DEVICE)

            opt.zero_grad()

            logits, c_loss = model(feats, tabular)

            cls = ce(logits, labels)

            loss = cls + params["lambda_contrast"] * c_loss

            loss.backward()
            opt.step()

        val_loss, bal_acc, auc = evaluate(
            model,
            val_loader,
            params["lambda_contrast"]
        )

        if bal_acc > best_score:
            best_score = bal_acc

            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / f"trial_{trial.number}_fold_{fold_id}.pt"
            )

    return best_score


# =========================================================
# OBJECTIVE
# =========================================================
def objective(trial):

    params = {
        "input_dim": 1536,
        "embed_dim": trial.suggest_categorical(
            "embed_dim", [128, 256, 512]
        ),
        "dropout": trial.suggest_float(
            "dropout", 0.0, 0.5
        ),
        "tab_hidden": trial.suggest_categorical(
            "tab_hidden", [16, 32, 64]
        ),
        "temperature": trial.suggest_float(
            "temperature", 0.03, 0.2
        ),
        "lambda_contrast": trial.suggest_float(
            "lambda_contrast", 0.05, 1.0
        ),
        "batch_size": 1,
        "lr": trial.suggest_float(
            "lr", 1e-6, 5e-4, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-7, 1e-2, log=True
        ),
    }

    folds = make_folds(
        train_df,
        ARGS.target_col,
        ARGS.n_folds
    )

    scores = []

    for fold_id, (tr, va) in enumerate(folds, 1):

        score = train_fold(
            trial,
            fold_id,
            dataset,
            tr,
            va,
            params
        )

        scores.append(score)

    return float(np.mean(scores))


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    df = pd.read_csv(ARGS.metadata_csv)

    train_df = df[df["split"] == "train"].reset_index(drop=True)

    dataset = DatasetClass(
        train_df,
        ARGS.feature_dir,
        ARGS.target_col,
        ARGS.feature_cols
    )

    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=ARGS.n_trials)

    print("\nBest params:")
    print(study.best_params)

    print("\nBest score:")
    print(study.best_value)