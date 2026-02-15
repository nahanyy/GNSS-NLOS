
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


TRAIN_NORM_XLSX = os.path.join(BASE_DIR, "data", "GNSS_train_norm.xlsx")

LLM_JSONL_TXT = os.path.join(BASE_DIR, "llm_outputs", "llm_outputs.jsonl")

OUT_DIR = os.path.join(BASE_DIR, "outputs", "student_mlp_distill")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
EPOCHS = 80
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4

ALPHA = 0.2

CONF_TH = 0.6

SOFT_CLIP = 1e-3

feature_cols = [
    "Elevation",
    "Azimuth",
    "CNR",
    "Pr_Residual",
    "Pr_residual_Root_of_sum_of_square_error",
    "Pr_Residual_cigma",
]
label_col = "label"

PT_PATH = os.path.join(OUT_DIR, "student_mlp_distill.pt")
META_JSON = os.path.join(OUT_DIR, "model_meta.json")


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(
                    f"JSONL parse failed: {path} line {ln}\nContent: {line}\nError: {e}"
                )
    return rows


class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def metrics_from_logits(logits: np.ndarray, y_true: np.ndarray):
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = (prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    pre = precision_score(y_true, pred, zero_division=0)

    auc = None
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, prob)

    cm = confusion_matrix(y_true, pred)
    return acc, f1, rec, pre, auc, cm


df = pd.read_excel(TRAIN_NORM_XLSX)

need_cols = feature_cols + [label_col]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in training set: {missing}")

for c in need_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=need_cols).copy().reset_index(drop=True)
df["row_id_norm"] = np.arange(len(df), dtype=int)


llm_rows = read_jsonl(LLM_JSONL_TXT)
llm_df = pd.DataFrame(llm_rows)

need_llm = ["sample_id", "nlos_probability", "confidence"]
missing_llm = [c for c in need_llm if c not in llm_df.columns]
if missing_llm:
    raise ValueError(f"Missing fields in LLM output: {missing_llm}")

llm_df["sample_id"] = pd.to_numeric(llm_df["sample_id"], errors="coerce").astype(int)
llm_df["nlos_probability"] = pd.to_numeric(llm_df["nlos_probability"], errors="coerce")
llm_df["confidence"] = pd.to_numeric(llm_df["confidence"], errors="coerce")

llm_df["nlos_probability"] = llm_df["nlos_probability"].clip(SOFT_CLIP, 1.0 - SOFT_CLIP)
llm_df["confidence"] = llm_df["confidence"].clip(0.0, 1.0)

if len(llm_df) != len(df):
    raise ValueError(
        f"LLM outputs size mismatch: got {len(llm_df)} lines, but training set has {len(df)} samples.\n"
        "For full-sample distillation, llm_outputs.jsonl must contain one entry per training sample.\n"
        "Expected sample_id to cover 0..N-1."
    )


llm_df = llm_df.sort_values("sample_id").reset_index(drop=True)
expected_ids = np.arange(len(df), dtype=int)
if not np.array_equal(llm_df["sample_id"].values, expected_ids):
    raise ValueError(
        "sample_id in LLM outputs must be a complete sequence 0..N-1 for full-sample alignment."
    )

df["soft_q"] = llm_df["nlos_probability"].values.astype(np.float32)
df["soft_conf"] = llm_df["confidence"].values.astype(np.float32)

num_soft = int(df["soft_q"].notna().sum())
print(f"[INFO] Total samples = {len(df)}")
print(f"[INFO] Soft-labeled samples (from LLM) = {num_soft}")

distill_mask = df["soft_q"].notna() & (df["soft_conf"] >= CONF_TH)
print(f"[INFO] Distill-enabled samples (conf >= {CONF_TH}) = {int(distill_mask.sum())}")


X = df[feature_cols].values.astype(np.float32)
y = df[label_col].astype(int).values

q = df["soft_q"].values.astype(np.float32)
conf = df["soft_conf"].values.astype(np.float32)
mask = distill_mask.values.astype(bool)

X_train, X_val, y_train, y_val, q_train, q_val, conf_train, conf_val, m_train, m_val = train_test_split(
    X, y, q, conf, mask,
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(in_dim=X.shape[1]).to(device)

opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
bce_logits = nn.BCEWithLogitsLoss(reduction="none")


def run_epoch(Xn, yn, qn, cn, mn, train: bool):
    model.train(train)
    N = len(yn)
    idx = np.arange(N)
    if train:
        np.random.shuffle(idx)

    total_loss = 0.0
    total_hard = 0.0
    total_soft = 0.0
    total_soft_count = 0

    for s in range(0, N, BATCH_SIZE):
        batch = idx[s:s + BATCH_SIZE]
        xb = torch.from_numpy(Xn[batch]).to(device)
        yb = torch.from_numpy(yn[batch].astype(np.float32)).to(device)

        logits = model(xb)

        hard_loss_vec = bce_logits(logits, yb)
        hard_loss = hard_loss_vec.mean()

        mb = mn[batch]
        if mb.any():
            qb = torch.from_numpy(qn[batch][mb].astype(np.float32)).to(device)
            cb = torch.from_numpy(cn[batch][mb].astype(np.float32)).to(device)

            logits_m = logits[torch.from_numpy(np.where(mb)[0]).to(device)]
            soft_loss_vec = bce_logits(logits_m, qb)

            w = cb
            soft_loss = (soft_loss_vec * w).mean()
            soft_count = int(mb.sum())
        else:
            soft_loss = torch.tensor(0.0, device=device)
            soft_count = 0

        loss = (1.0 - ALPHA) * hard_loss + ALPHA * soft_loss

        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

        total_loss += float(loss.detach().cpu()) * len(batch)
        total_hard += float(hard_loss.detach().cpu()) * len(batch)
        total_soft += float(soft_loss.detach().cpu()) * len(batch)
        total_soft_count += soft_count

    return total_loss / N, total_hard / N, total_soft / N, total_soft_count


best = {"auc": -1, "epoch": -1, "metrics": None, "state": None}

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_hard, tr_soft, tr_soft_n = run_epoch(X_train, y_train, q_train, conf_train, m_train, train=True)
    va_loss, va_hard, va_soft, va_soft_n = run_epoch(X_val, y_val, q_val, conf_val, m_val, train=False)

    with torch.no_grad():
        xb = torch.from_numpy(X_val).to(device)
        logits_val = model(xb).detach().cpu().numpy()

    acc, f1, rec, pre, auc, cm = metrics_from_logits(logits_val, y_val)
    score = acc if auc is None else auc

    if (auc is not None and auc > best["auc"]) or (auc is None and acc > (best["metrics"]["acc"] if best["metrics"] else -1)):
        best["auc"] = -1 if auc is None else float(auc)
        best["epoch"] = epoch
        best["metrics"] = {
            "acc": float(acc),
            "f1": float(f1),
            "recall": float(rec),
            "precision": float(pre),
            "auc": None if auc is None else float(auc),
            "confusion_matrix": cm.tolist(),
            "val_loss": float(va_loss),
        }
        best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"[E{epoch:03d}] train_loss={tr_loss:.4f} (hard={tr_hard:.4f}, soft={tr_soft:.4f}, softN={tr_soft_n}) | "
            f"val_auc={auc if auc is not None else 'N/A'} acc={acc:.4f}"
        )

model.load_state_dict(best["state"])

print("\n========== Best Validation (MLP Distill - Full) ==========")
print({"epoch": best["epoch"], **best["metrics"]})


torch.save(model.state_dict(), PT_PATH)

meta = {
    "model_type": "PyTorch_MLP_Distill_Full",
    "feature_cols": feature_cols,
    "label_col": label_col,
    "train_file": os.path.relpath(TRAIN_NORM_XLSX, BASE_DIR),
    "llm_output_file": os.path.relpath(LLM_JSONL_TXT, BASE_DIR),
    "random_state": RANDOM_STATE,
    "train_settings": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "alpha": ALPHA,
        "conf_threshold": CONF_TH,
        "soft_clip": SOFT_CLIP,
        "note": "Hard BCE on all samples; soft BCE on all LLM-labeled samples with confidence gating.",
    },
    "data_stats": {
        "total_samples": int(len(df)),
        "soft_labeled_samples": int(num_soft),
        "distill_enabled_samples": int(distill_mask.sum()),
    },
    "best_val": best["metrics"],
}

with open(META_JSON, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n[OK] Model saved to:", PT_PATH)
print("[OK] Meta  saved to:", META_JSON)
