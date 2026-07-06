"""
==============================================================================
Temporal Heterogeneous Graph Neural Network— CIC-IDS2018 Edition
==============================================================================

ENABLED COMPONENTS :
  ✓ Temporal Graph Memory (GRU + half-life decay)
  ✓ Memory-Aware Walk Generator (ε-exploration, memory-biased sampling)
  ✓ Heterogeneous Edge Convolution (3 layers, 8 heads, gated residual)
  ✓ Contrastive Learning (InfoNCE with hard-negative queue)
  ✓ Path Reconstruction (next-node prediction)
  ✓ Learnable Rarity Scoring (online EMA + Mahalanobis MLP)
  ✓ Temporal Consistency (smooth embedding loss)
  ✓ Transformer Walk Encoder (self-attention)
  ✓ MITRE ATT&CK Mapping

ABLATION STUDY CONFIGURATIONS:
  - full: All components enabled (baseline)
  - no_contrastive: Remove contrastive learning
  - no_temporal: Remove temporal consistency loss
  - no_path_recon: Remove path reconstruction
  - no_rarity: Remove rarity scoring
  - no_memory: Replace memory with zeros
  - recon_only: Only reconstruction loss
  - no_hard_negatives: Standard InfoNCE (no hard negatives)
  - no_edge_types: All edges treated equally
  - walk_gru: GRU instead of Transformer for walks
  - single_hec: Single HEC layer vs 3 layers

DATASET NOTES — CIC-IDS2018:
  • No source IP column → node identity = service endpoint (dst_port/protocol)
    for destination nodes + behavioral fingerprint for source nodes.
  • Proxy attack labels (Infiltration, FTP-BruteForce, SSH-BruteForce) are used
    ONLY for final evaluation reporting, NEVER during training.
  • Numeric feature set mirrors CIC-IDS2018 flow statistics (16 features).

==============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Imports & reproducibility
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, math, random, warnings, logging, copy, json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import (
    average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Configuration (with Ablation Support)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── Dataset ──────────────────────────────────────────────────────────────
    data_path:   str            = "data/cic_ids2018"
    max_files:   Optional[int]  = None
    max_rows:    Optional[int]  = None
    sample_frac: Optional[float]= None

    temporal_window_s: float = 300.0
    train_ratio:       float = 0.70
    val_ratio:         float = 0.15

    # ── Graph / feature dims ─────────────────────────────────────────────────
    node_feat_dim:      int = 11   # see build_node_features_window
    edge_feat_dim:      int = 16   # CIC numeric features (padded to this)
    protocol_emb_dim:   int = 4
    port_emb_dim:       int = 8
    port_cat_emb_dim:   int = 4
    edge_type_emb_dim:  int = 8
    num_edge_types:     int = 5    # normal / sensitive / frequent / scan / crypto-like
    num_proto:          int = 4    # placeholder; overwritten from data

    # ── Full Architecture settings ───────────────────────────────────────────
    hidden_dim:              int   = 128
    num_hec_layers:          int   = 3
    num_heads:               int   = 8
    dropout:                 float = 0.1
    memory_dim:              int   = 128
    memory_decay_half_life:  float = 3600.0

    # ── Walk settings ────────────────────────────────────────────────────────
    walk_length:              int   = 4
    num_walks:                int   = 20
    walk_temporal_window_s:   float = 600.0
    walk_encoder:             str   = "transformer"
    walk_encoder_layers:      int   = 2
    walk_encoder_heads:       int   = 8
    walk_exploration_epsilon: float = 0.1

    # ── Loss weights (all enabled) ───────────────────────────────────────────
    lambda_recon:       float = 1.0
    lambda_contrastive: float = 0.5
    lambda_temporal:    float = 0.3
    lambda_path_recon:  float = 0.4
    lambda_rarity:      float = 0.2
    temperature:        float = 0.07
    hard_neg_ratio:     float = 0.5
    hard_neg_queue_size:int   = 10000

    # ── Path scoring weights ─────────────────────────────────────────────────
    path_score_w_recon:      float = 0.4
    path_score_w_rarity:     float = 0.4
    path_score_w_likelihood: float = 0.2

    # ── Training ─────────────────────────────────────────────────────────────
    epochs:          int   = 30
    lr:              float = 1e-4
    weight_decay:    float = 1e-4
    batch_size:      int   = 256
    patience:        int   = 10
    grad_clip:       float = 1.0
    mixed_precision: bool  = False

    # ── Labels ───────────────────────────────────────────────────────────────
    proxy_attack_labels: List[str] = field(
        default_factory=lambda: ["Infiltration", "FTP-BruteForce", "SSH-BruteForce"]
    )
    sensitive_ports: List[int] = field(
        default_factory=lambda: [21, 22, 23, 25, 445, 3389, 5900, 1433, 3306]
    )
    
    # ── Ablation Study Settings ──────────────────────────────────────────────
    ablation_config: str = "full"  # Options defined in ABLATION_CONFIGS
    ablation_seeds: int = 3  # Number of random seeds to run per configuration
    ablation_output_dir: str = "ablation_results_cic"


# ─────────────────────────────────────────────────────────────────────────────
# 1.1 Ablation Configuration Definitions
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS = {
    "full": {
        "name": "Full Model (All Components)",
        "description": "All losses and components enabled",
        "flags": {
            "use_contrastive": True,
            "use_temporal": True,
            "use_path_recon": True,
            "use_rarity": True,
            "use_memory": True,
            "use_hard_negatives": True,
            "use_edge_types": True,
            "walk_encoder": "transformer",
            "num_hec_layers": 3,
        }
    },
    "no_contrastive": {
        "name": "No Contrastive Learning",
        "description": "Remove InfoNCE contrastive loss",
        "flags": {
            "use_contrastive": False,
            "use_temporal": True,
            "use_path_recon": True,
            "use_rarity": True,
            "use_memory": True,
            "use_hard_negatives": True,
            "use_edge_types": True,
            "walk_encoder": "transformer",
            "num_hec_layers": 3,
        }
    },
    "no_temporal": {
        "name": "No Temporal Consistency",
        "description": "Remove temporal consistency loss",
        "flags": {
            "use_contrastive": True,
            "use_temporal": False,
            "use_path_recon": True,
            "use_rarity": True,
            "use_memory": True,
            "use_hard_negatives": True,
            "use_edge_types": True,
            "walk_encoder": "transformer",
            "num_hec_layers": 3,
        }
    },
    "no_path_recon": {
        "name": "No Path Reconstruction",
        "description": "Remove path reconstruction loss",
        "flags": {
            "use_contrastive": True,
            "use_temporal": True,
            "use_path_recon": False,
            "use_rarity": True,
            "use_memory": True,
            "use_hard_negatives": True,
            "use_edge_types": True,
            "walk_encoder": "transformer",
            "num_hec_layers": 3,
        }
    },
    "no_rarity": {
        "name": "No Rarity Scoring",
        "description": "Remove learnable rarity scorer",
        "flags": {
            "use_contrastive": True,
            "use_temporal": True,
            "use_path_recon": True,
            "use_rarity": False,
            "use_memory": True,
            "use_hard_negatives": True,
            "use_edge_types": True,
            "walk_encoder": "transformer",
            "num_hec_layers": 3,
        }
    },
    "no_memory": {
        "name": "No Temporal Memory",
        "description": "Replace GRU memory with zeros",
        "flags": {
            "use_contrastive": True,
            "use_temporal": True,
            "use_path_recon": True,
            "use_rarity": True,
            "use_memory": False,
            "use_hard_negatives": True,
            "use_edge_types": True,
            "walk_encoder": "transformer",
            "num_hec_layers": 3,
        }
    },
    "recon_only": {
        "name": "Reconstruction Only",
        "description": "Only edge reconstruction loss",
        "flags": {
            "use_contrastive": False,
            "use_temporal": False,
            "use_path_recon": False,
            "use_rarity": False,
            "use_memory": True,
            "use_hard_negatives": True,
            "use_edge_types": True,
            "walk_encoder": "transformer",
            "num_hec_layers": 3,
        }
    },
    "no_hard_negatives": {
        "name": "No Hard Negatives (Standard InfoNCE)",
        "description": "Remove hard negative mining from contrastive loss",
        "flags": {
            "use_contrastive": True,
            "use_temporal": True,
            "use_path_recon": True,
            "use_rarity": True,
            "use_memory": True,
            "use_hard_negatives": False,
            "use_edge_types": True,
            "walk_encoder": "transformer",
            "num_hec_layers": 3,
        }
    },
    "no_edge_types": {
        "name": "No Edge Types (Homogeneous)",
        "description": "All edges treated as same type",
        "flags": {
            "use_contrastive": True,
            "use_temporal": True,
            "use_path_recon": True,
            "use_rarity": True,
            "use_memory": True,
            "use_hard_negatives": True,
            "use_edge_types": False,
            "walk_encoder": "transformer",
            "num_hec_layers": 3,
        }
    },
    "walk_gru": {
        "name": "GRU Walk Encoder",
        "description": "Replace Transformer with GRU for walk encoding",
        "flags": {
            "use_contrastive": True,
            "use_temporal": True,
            "use_path_recon": True,
            "use_rarity": True,
            "use_memory": True,
            "use_hard_negatives": True,
            "use_edge_types": True,
            "walk_encoder": "gru",
            "num_hec_layers": 3,
        }
    },
    "single_hec": {
        "name": "Single HEC Layer",
        "description": "Only one Heterogeneous Edge Convolution layer",
        "flags": {
            "use_contrastive": True,
            "use_temporal": True,
            "use_path_recon": True,
            "use_rarity": True,
            "use_memory": True,
            "use_hard_negatives": True,
            "use_edge_types": True,
            "walk_encoder": "transformer",
            "num_hec_layers": 1,
        }
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CIC-IDS2018 column map & numeric feature list
# ─────────────────────────────────────────────────────────────────────────────

# Canonical column names used internally (mapped from raw CIC headers)
_CIC_RENAME: Dict[str, str] = {
    "Dst Port":        "dst_port",
    "Protocol":        "protocol",
    "Timestamp":       "timestamp",
    "Flow Duration":   "duration",
    "Tot Fwd Pkts":    "fwd_pkts",
    "Tot Bwd Pkts":    "bwd_pkts",
    "TotLen Fwd Pkts": "fwd_bytes",
    "TotLen Bwd Pkts": "bwd_bytes",
    "Flow IAT Mean":   "iat_mean",
    "Flow IAT Std":    "iat_std",
    "Active Mean":     "active_mean",
    "Idle Mean":       "idle_mean",
    "SYN Flag Cnt":    "syn_cnt",
    "FIN Flag Cnt":    "fin_cnt",
    "RST Flag Cnt":    "rst_cnt",
    "Label":           "label",
}

# The 12 raw numeric columns available in CIC-IDS2018 flows
_CIC_NUMERIC_COLS: List[str] = [
    "duration", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
    "iat_mean", "iat_std", "active_mean", "idle_mean",
    "syn_cnt", "fin_cnt", "rst_cnt",
]

SENSITIVE_PORTS: Set[int] = {21, 22, 23, 25, 445, 3389, 5900, 1433, 3306}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Data Loading & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _port_bucket(port: int) -> int:
    if port < 1024:  return 0
    if port < 49152: return 1
    return 2


def _discover_csv_files(data_path: str) -> List[str]:
    if os.path.isfile(data_path):
        return [data_path]
    if os.path.isdir(data_path):
        files = sorted(
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.lower().endswith(".csv")
        )
        if not files:
            raise FileNotFoundError(f"No CSV files in: {data_path}")
        return files
    raise FileNotFoundError(f"data_path not found: {data_path}")


def _build_service_nodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    CIC-IDS2018 has no source IP column.
    Destination node : service endpoint  "svc:<dst_port>/p<protocol>"
    Source node      : behavioral bucket "behav:<proto*100 + pkt_bin*10 + byte_bin>"
    """
    df["dst_node"] = "svc:" + df["dst_port"].astype(str) + "/p" + df["protocol"].astype(str)

    proto_int = pd.to_numeric(df["protocol"], errors="coerce").fillna(0).astype(int)
    fwd_pkts  = pd.to_numeric(df["fwd_pkts"], errors="coerce").fillna(0).replace([np.inf, -np.inf], 0)
    fwd_bytes = pd.to_numeric(df["fwd_bytes"], errors="coerce").fillna(0).replace([np.inf, -np.inf], 0)

    def _qbin(s, q=10):
        try:
            if s.nunique() > 1:
                return pd.qcut(s.clip(0, s.quantile(0.99)), q=q, labels=False, duplicates="drop")
        except Exception:
            pass
        return (s // max(s.max() / q, 1)).clip(0, q - 1)

    pkt_bin  = _qbin(fwd_pkts).fillna(0).astype(int)
    byte_bin = _qbin(fwd_bytes).fillna(0).astype(int)
    bucket   = proto_int * 100 + pkt_bin * 10 + byte_bin
    df["src_node"] = "behav:" + bucket.astype(str)

    # Alias columns expected by downstream code
    df["src_ip"] = df["src_node"]
    df["dst_ip"] = df["dst_node"]
    df["src_port"] = 0
    return df


def _read_csv_limited(path: str, rows_remaining: int,
                      sample_frac: Optional[float]) -> pd.DataFrame:
    if rows_remaining <= 0:
        return pd.DataFrame()
    try:
        if sample_frac and 0.0 < sample_frac < 1.0:
            chunks = []
            for ch in pd.read_csv(path, low_memory=False, chunksize=200_000,
                                   on_bad_lines="skip"):
                ch.columns = ch.columns.str.strip()
                ch.rename(columns=_CIC_RENAME, inplace=True)
                ch = ch.sample(frac=sample_frac, random_state=42)
                chunks.append(ch)
                if sum(len(c) for c in chunks) >= rows_remaining:
                    break
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(path, low_memory=False, nrows=rows_remaining,
                              on_bad_lines="skip")
            df.columns = df.columns.str.strip()
            df.rename(columns=_CIC_RENAME, inplace=True)
    except Exception as e:
        logger.warning("Read error %s: %s", path, e)
        return pd.DataFrame()
    return df.head(rows_remaining)


def load_and_preprocess(cfg: Config) -> Tuple[pd.DataFrame, Dict]:
    all_files = _discover_csv_files(cfg.data_path)
    logger.info("Found %d CSV file(s)", len(all_files))
    if cfg.max_files:
        all_files = all_files[:cfg.max_files]

    budget, consumed, frames = cfg.max_rows or 10_000_000_000, 0, []
    for fpath in all_files:
        if consumed >= budget:
            break
        logger.info("  Reading %s", os.path.basename(fpath))
        chunk = _read_csv_limited(fpath, budget - consumed, cfg.sample_frac)
        if len(chunk) == 0:
            continue
        frames.append(chunk)
        consumed += len(chunk)
        logger.info("    → %d rows (total %d)", len(chunk), consumed)

    if not frames:
        raise RuntimeError("No CIC-IDS2018 data loaded.")

    df = pd.concat(frames, ignore_index=True)
    logger.info("Raw concat: %s", df.shape)

    # ── Numeric cleaning ─────────────────────────────────────────────────────
    for col in _CIC_NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        df[col] = df[col].replace([np.inf, -np.inf], 0.0)

    # Log-transform right-skewed features
    for col in ["fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "duration"]:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # ── Build service/behavioral nodes ───────────────────────────────────────
    for col in ["dst_port", "protocol"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df = _build_service_nodes(df)

    # ── Timestamp ────────────────────────────────────────────────────────────
    parsed = None
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
                "%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", None):
        try:
            kw = {"format": fmt} if fmt else {"infer_datetime_format": True}
            cand = pd.to_datetime(df["timestamp"].astype(str), errors="coerce", **kw)
            if cand.notna().mean() > 0.5:
                parsed = cand
                break
        except Exception:
            continue
    if parsed is None:
        num_ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if num_ts.notna().mean() > 0.5:
            parsed = pd.to_datetime(num_ts, unit="s", errors="coerce")
        else:
            raise RuntimeError(f"Cannot parse timestamps. Sample: {df['timestamp'].head(3).tolist()}")

    df["timestamp"] = parsed
    df.dropna(subset=["timestamp"], inplace=True)
    df["ts_unix"] = df["timestamp"].astype(np.int64) / 1_000_000_000
    df.sort_values("ts_unix", inplace=True, ignore_index=True)

    # ── Labels ───────────────────────────────────────────────────────────────
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip()
        df["is_attack"] = df["label"].isin(cfg.proxy_attack_labels).astype(int)
    else:
        df["label"] = "Normal"
        df["is_attack"] = 0
    logger.info("Label dist:\n%s", df["label"].value_counts().to_string())

    # ── Port / protocol encodings ────────────────────────────────────────────
    df["dst_port"]        = pd.to_numeric(df["dst_port"], errors="coerce").fillna(0).astype(int)
    df["src_port"]        = 0
    df["src_port_cat"]    = 0
    df["dst_port_cat"]    = df["dst_port"].apply(_port_bucket)
    df["is_sensitive_dst"]= df["dst_port"].isin(SENSITIVE_PORTS).astype(int)

    proto_vals = sorted(df["protocol"].unique().tolist())
    proto_map  = {p: i for i, p in enumerate(proto_vals)}
    df["protocol_id"] = df["protocol"].map(proto_map).fillna(0).astype(int)

    # ── Node indexing ────────────────────────────────────────────────────────
    all_nodes = pd.concat([df["src_node"], df["dst_node"]]).unique()
    node2id   = {n: i for i, n in enumerate(sorted(all_nodes))}
    df["src_id"] = df["src_node"].map(node2id)
    df["dst_id"] = df["dst_node"].map(node2id)
    num_nodes  = len(node2id)
    logger.info("Nodes: %d (service + behavioral) | Flows: %d", num_nodes, len(df))

    # ── Edge-type assignment ─────────────────────────────────────────────────
    pair_cnt       = df.groupby(["src_id", "dst_id"]).size()
    df["pair_count"] = df.set_index(["src_id", "dst_id"]).index.map(pair_cnt).values
    df["edge_type"]  = 0
    df.loc[df["is_sensitive_dst"] == 1,                              "edge_type"] = 1
    df.loc[df["pair_count"] >= 5,                                    "edge_type"] = 2
    df.loc[(df["syn_cnt"] > 10) & (df["fin_cnt"] == 0),             "edge_type"] = 3
    # Heuristic "crypto-like": high byte-to-packet ratio after log1p
    df.loc[(df["fwd_bytes"] - df["fwd_pkts"]) > 2.0,                "edge_type"] = 4

    # ── Feature scaling (MinMax, fits on train split only) ───────────────────
    train_cutoff = df["ts_unix"].quantile(cfg.train_ratio)
    df_train     = df[df["ts_unix"] <= train_cutoff]
    scaler       = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_train[_CIC_NUMERIC_COLS].values)
    df[_CIC_NUMERIC_COLS] = scaler.transform(df[_CIC_NUMERIC_COLS].values).clip(0, 1)
    logger.info("Feature range after scaling: min=%.4f  max=%.4f",
                df[_CIC_NUMERIC_COLS].min().min(), df[_CIC_NUMERIC_COLS].max().max())

    meta = {
        "ip2id":    node2id,
        "id2ip":    {v: k for k, v in node2id.items()},
        "num_nodes":num_nodes,
        "proto_map":proto_map,
        "num_proto":len(proto_map),
        "scaler":   scaler,
    }
    return df, meta


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Node Feature Engineering (window-level)
# ─────────────────────────────────────────────────────────────────────────────

def build_node_features_window(df_window: pd.DataFrame, num_nodes: int) -> torch.Tensor:
    """
    11-dimensional node feature vector computed per temporal window.

    Idx  Feature
    ---  -------
    0    total degree (flows as src + dst)
    1    in-degree  (flows as dst)
    2    out-degree (flows as src)
    3    unique destination nodes
    4    unique source nodes
    5    flow rate (flows / window span)
    6    mean flow duration
    7    mean fwd bytes
    8    sensitive-port ratio
    9    syn-scan indicator (syn>10 & fin==0 fraction)
    10   IAT burstiness (std/mean, clipped)
    """
    feats = np.zeros((num_nodes, 11), dtype=np.float32)
    if len(df_window) == 0:
        return torch.tensor(feats, dtype=torch.float32)
    span = df_window["ts_unix"].max() - df_window["ts_unix"].min() + 1e-6

    for sid, grp in df_window.groupby("src_id"):
        n = len(grp)
        feats[sid, 0] += n
        feats[sid, 2] += n
        feats[sid, 3]  = grp["dst_id"].nunique()
        feats[sid, 5]  = n / span
        feats[sid, 6]  = grp["duration"].mean()
        feats[sid, 7]  = grp["fwd_bytes"].mean()
        feats[sid, 8]  = grp["is_sensitive_dst"].mean()
        feats[sid, 9]  = ((grp["syn_cnt"] > 10) & (grp["fin_cnt"] == 0)).mean() \
                          if "syn_cnt" in grp.columns else 0.0
        iat_m = grp["iat_mean"].mean() if "iat_mean" in grp.columns else 0.0
        iat_s = grp["iat_std"].mean()  if "iat_std"  in grp.columns else 0.0
        feats[sid, 10] = float(np.clip(iat_s / (iat_m + 1e-6), 0, 10))

    for did, grp in df_window.groupby("dst_id"):
        n = len(grp)
        feats[did, 0] += n
        feats[did, 1] += n
        feats[did, 4]  = grp["src_id"].nunique()

    mu    = feats.mean(0, keepdims=True)
    sigma = feats.std(0,  keepdims=True) + 1e-8
    return torch.tensor((feats - mu) / sigma, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Temporal Node Memory (with ablation support)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalNodeMemory(nn.Module):
    def __init__(self, num_nodes: int, memory_dim: int,
                 edge_feat_dim: int, decay_half_life: float = 3600.0,
                 disabled: bool = False):
        super().__init__()
        self.disabled = disabled
        self.decay_rate = math.log(2.0) / max(decay_half_life, 1.0)
        self.gru        = nn.GRUCell(memory_dim, memory_dim)
        self.msg_proj   = nn.Linear(memory_dim * 2 + edge_feat_dim, memory_dim)
        self.register_buffer("memory",      torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(num_nodes))
        self.eps = 1e-8

    def reset(self):
        if not self.disabled:
            self.memory.zero_()
            self.last_update.zero_()

    def _decay(self, ids: torch.Tensor, ts: float) -> torch.Tensor:
        if self.disabled:
            return torch.ones(len(ids), 1, device=self.memory.device)
        dt = (ts - self.last_update[ids]).clamp(0.0, 1e6)
        return torch.exp(-dt * self.decay_rate).unsqueeze(-1)

    def get(self, ids: torch.Tensor, ts: float) -> torch.Tensor:
        if self.disabled:
            return torch.zeros(len(ids), self.memory.size(1), device=self.memory.device)
        mem = self.memory[ids] * self._decay(ids, ts)
        return torch.nan_to_num(mem, nan=0.0)

    def update(self, src: torch.Tensor, dst: torch.Tensor,
               edge_feat: torch.Tensor, ts: float):
        if self.disabled:
            return
        with torch.no_grad():
            N      = self.memory.size(0)
            ef_dim = edge_feat.size(-1)
            device = edge_feat.device
            edge_feat = torch.nan_to_num(edge_feat, nan=0.0)

            ef_sum = torch.zeros(N, ef_dim, device=device)
            ef_cnt = torch.zeros(N, 1,      device=device)
            ef_sum.scatter_add_(0, src.unsqueeze(-1).expand_as(edge_feat), edge_feat)
            ef_cnt.scatter_add_(0, src.unsqueeze(-1), torch.ones(src.size(0), 1, device=device))
            ef_sum.scatter_add_(0, dst.unsqueeze(-1).expand_as(edge_feat), edge_feat)
            ef_cnt.scatter_add_(0, dst.unsqueeze(-1), torch.ones(dst.size(0), 1, device=device))

            involved = torch.unique(torch.cat([src, dst]))
            s_mem    = self.get(involved, ts)
            ef_agg   = (ef_sum / (ef_cnt + self.eps))[involved]

            msg     = torch.nan_to_num(self.msg_proj(torch.cat([s_mem, s_mem, ef_agg], -1)), nan=0.0)
            new_mem = torch.nan_to_num(self.gru(msg, s_mem), nan=0.0)
            self.memory[involved]      = new_mem
            self.last_update[involved] = ts


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Categorical Embedder (with ablation support)
# ─────────────────────────────────────────────────────────────────────────────

class CategoricalEmbedder(nn.Module):
    """
    CIC-IDS2018 variant — no per-flow crypto/isint flag columns.
    Embeds: protocol, src_port, dst_port, dst_port_category, edge_type.
    """
    def __init__(self, cfg: Config, num_proto: int, use_edge_types: bool = True):
        super().__init__()
        self.use_edge_types = use_edge_types
        self.proto_emb    = nn.Embedding(num_proto + 1,      cfg.protocol_emb_dim)
        self.src_port_emb = nn.Embedding(65536,               cfg.port_emb_dim)
        self.dst_port_emb = nn.Embedding(65536,               cfg.port_emb_dim)
        self.port_cat_emb = nn.Embedding(3,                   cfg.port_cat_emb_dim)
        if use_edge_types:
            self.edge_type_emb = nn.Embedding(cfg.num_edge_types,  cfg.edge_type_emb_dim)
        else:
            self.edge_type_emb = None
            
        self.out_dim = (cfg.protocol_emb_dim + 2 * cfg.port_emb_dim
                        + cfg.port_cat_emb_dim)
        if use_edge_types:
            self.out_dim += cfg.edge_type_emb_dim

    def forward(self, proto_ids, src_ports, dst_ports,
                dst_port_cats, edge_types) -> torch.Tensor:
        components = [
            self.proto_emb(proto_ids),
            self.src_port_emb(src_ports.clamp(0, 65535)),
            self.dst_port_emb(dst_ports.clamp(0, 65535)),
            self.port_cat_emb(dst_port_cats),
        ]
        if self.use_edge_types and self.edge_type_emb is not None:
            components.append(self.edge_type_emb(edge_types))
        return torch.cat(components, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Heterogeneous Edge Convolution (with ablation support)
# ─────────────────────────────────────────────────────────────────────────────

class HeterogeneousEdgeConv(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int,
                 num_heads: int, num_edge_types: int, dropout: float = 0.1,
                 use_edge_types: bool = True):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.H  = hidden_dim
        self.nh = num_heads
        self.dh = hidden_dim // num_heads
        self.num_edge_types = num_edge_types
        self.use_edge_types = use_edge_types

        if use_edge_types:
            self.W_msg = nn.ModuleList([
                nn.Linear(node_dim + edge_dim, hidden_dim, bias=False)
                for _ in range(num_edge_types)
            ])
        else:
            self.W_msg = nn.Linear(node_dim + edge_dim, hidden_dim, bias=False)
            
        self.W_q     = nn.Linear(node_dim,  hidden_dim, bias=False)
        self.W_k     = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o     = nn.Linear(hidden_dim, hidden_dim)
        self.res_proj= (nn.Linear(node_dim, hidden_dim, bias=False)
                        if node_dim != hidden_dim else nn.Identity())
        self.gate    = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.temporal_gamma = nn.Parameter(torch.tensor(0.1))
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.act        = nn.GELU()
        self.eps = 1e-8

    def forward(self, h, edge_index, edge_feat, edge_type, delta_t):
        N, E = h.size(0), edge_index.size(1)
        src, dst = edge_index[0], edge_index[1]
        h         = torch.nan_to_num(h,         nan=0.0)
        edge_feat = torch.nan_to_num(edge_feat, nan=0.0)

        src_feats = h[src]
        msgs = torch.zeros(E, self.H, device=h.device)
        
        if self.use_edge_types:
            for r in range(self.num_edge_types):
                mask = (edge_type == r)
                if mask.any():
                    inp = torch.cat([src_feats[mask], edge_feat[mask]], dim=-1)
                    msgs[mask] = self.W_msg[r](torch.nan_to_num(inp, nan=0.0))
        else:
            inp = torch.cat([src_feats, edge_feat], dim=-1)
            msgs = self.W_msg(torch.nan_to_num(inp, nan=0.0))

        Q = self.W_q(h[dst]).view(E, self.nh, self.dh)
        K = self.W_k(msgs).view(E, self.nh, self.dh)
        attn = (Q * K).sum(-1) / math.sqrt(self.dh + self.eps)

        if delta_t is not None:
            delta_t = torch.nan_to_num(delta_t, nan=0.0)
            attn = attn + self.temporal_gamma * (-delta_t.unsqueeze(-1).clamp(-1e6, 1e6))

        attn = attn - attn.max(dim=0, keepdim=True)[0]
        attn_exp = torch.exp(attn.clamp(-50, 50))
        agg_exp  = torch.zeros(N, self.nh, device=h.device)
        agg_exp.scatter_add_(0, dst.unsqueeze(-1).expand_as(attn_exp), attn_exp)
        norm   = agg_exp[dst] + self.eps
        attn_w = (attn_exp / norm).unsqueeze(-1)

        weighted = (attn_w * msgs.view(E, self.nh, self.dh)).view(E, self.H)
        agg      = torch.zeros(N, self.H, device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted), weighted)

        h_res    = self.res_proj(h)
        out      = self.act(self.W_o(agg))
        gate_w   = self.gate(torch.nan_to_num(torch.cat([h_res, out], -1), nan=0.0))
        out      = gate_w * out + (1 - gate_w) * h_res
        out      = self.dropout(self.layer_norm(out))
        return torch.nan_to_num(out, nan=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Memory-Aware Walk Generator (with ablation support)
# ─────────────────────────────────────────────────────────────────────────────

class MemoryAwareWalkGenerator:
    def __init__(self, walk_length: int = 4, num_walks: int = 20,
                 temporal_window: float = 600.0, exploration_epsilon: float = 0.1):
        self.walk_length        = walk_length
        self.num_walks          = num_walks
        self.temporal_window    = temporal_window
        self.exploration_epsilon= exploration_epsilon

    def build_adjacency(self, df: pd.DataFrame) -> Dict:
        adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for row in df[["src_id", "dst_id", "ts_unix"]].itertuples(index=False):
            adj[row.src_id].append((row.dst_id, row.ts_unix))
        for k in adj:
            adj[k].sort(key=lambda x: x[1])
        return adj

    def generate_walks(self, adj: Dict,
                       memory: Optional[TemporalNodeMemory] = None,
                       current_ts: float = 0.0) -> List[Dict]:
        walks = []
        for src, src_events in adj.items():
            if not src_events:
                continue
            for _ in range(self.num_walks):
                seed_dst, seed_ts = random.choice(src_events)
                walk = [(src, seed_ts), (seed_dst, seed_ts)]
                cur_node, cur_ts = seed_dst, seed_ts
                for _ in range(self.walk_length - 1):
                    candidates = [(n, t) for n, t in adj.get(cur_node, [])
                                  if cur_ts < t <= cur_ts + self.temporal_window]
                    if not candidates:
                        break
                    if random.random() < self.exploration_epsilon:
                        cur_node, cur_ts = random.choice(candidates)
                    else:
                        weights = []
                        for n, t in candidates:
                            dt_decay  = math.exp(-(t - cur_ts) / 300.0)
                            mem_norm  = 1.0
                            if memory is not None and not memory.disabled:
                                nid      = torch.tensor([n], dtype=torch.long, device=DEVICE)
                                mem_norm = 1.0 + float(memory.get(nid, current_ts).norm().item())
                            weights.append(dt_decay * mem_norm)
                        total  = sum(weights) + 1e-8
                        weights= [w / total for w in weights]
                        idx    = np.random.choice(len(candidates), p=weights)
                        cur_node, cur_ts = candidates[idx]
                    walk.append((cur_node, cur_ts))
                if len(walk) == self.walk_length + 1:
                    walks.append({"source": src, "walk": walk})
        return walks


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Walk Encoder (Transformer/GRU with ablation support)
# ─────────────────────────────────────────────────────────────────────────────

class WalkEncoder(nn.Module):
    def __init__(self, hidden_dim: int, encoder_type: str = "transformer",
                 num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.encoder_type = encoder_type
        self.input_proj   = nn.Linear(hidden_dim + 1, hidden_dim)
        if encoder_type == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        else:
            self.encoder = nn.GRU(hidden_dim, hidden_dim, num_layers,
                                   batch_first=True, dropout=dropout)
        self.pool = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, walk_embs: torch.Tensor, walk_ts: torch.Tensor) -> torch.Tensor:
        walk_embs = torch.nan_to_num(walk_embs, nan=0.0)
        walk_ts   = torch.nan_to_num(walk_ts,   nan=0.0)
        x = self.input_proj(torch.cat([walk_embs, walk_ts.unsqueeze(-1)], -1))
        x = torch.nan_to_num(x, nan=0.0)
        if self.encoder_type == "transformer":
            out = self.pool(self.encoder(x).mean(1))
        else:
            _, h = self.encoder(x)
            out  = self.pool(h[-1])
        return torch.nan_to_num(out, nan=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Path Reconstruction Head
# ─────────────────────────────────────────────────────────────────────────────

class PathReconstructionHead(nn.Module):
    def __init__(self, hidden_dim: int, num_nodes: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, num_nodes),
        )

    def forward(self, step_embeddings: torch.Tensor) -> torch.Tensor:
        if step_embeddings.size(1) < 2:
            return torch.zeros(step_embeddings.size(0), 1,
                                step_embeddings.size(2), device=step_embeddings.device)
        step_embeddings = torch.nan_to_num(step_embeddings, nan=0.0)
        out = self.predictor(step_embeddings[:, :-1, :])
        return torch.nan_to_num(out, nan=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Hard-Negative InfoNCE (with ablation support)
# ─────────────────────────────────────────────────────────────────────────────

class HardNegativeInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07, hard_neg_ratio: float = 0.5,
                 queue_size: int = 10000, embedding_dim: int = 128,
                 use_hard_negatives: bool = True):
        super().__init__()
        self.tau   = max(temperature, 0.01)
        self.hnr   = hard_neg_ratio if use_hard_negatives else 0.0
        self.use_hard_negatives = use_hard_negatives
        self.queue_size = queue_size
        self.register_buffer("queue_emb", torch.randn(queue_size, embedding_dim))
        self.register_buffer("queue_src", torch.full((queue_size,), -1, dtype=torch.long))
        self.queue_ptr    = 0
        self.queue_filled = 0
        self.eps = 1e-8

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor, source_ids: torch.Tensor):
        embeddings = torch.nan_to_num(embeddings, nan=0.0)
        B   = embeddings.size(0)
        end = min(self.queue_ptr + B, self.queue_size)
        w   = end - self.queue_ptr
        self.queue_emb[self.queue_ptr:end] = embeddings[:w]
        self.queue_src[self.queue_ptr:end] = source_ids[:w]
        self.queue_ptr    = end % self.queue_size
        self.queue_filled = min(self.queue_filled + w, self.queue_size)

    def forward(self, z: torch.Tensor, source_ids: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z.device)
        z      = torch.nan_to_num(z, nan=0.0)
        z_norm = F.normalize(z, dim=-1, eps=self.eps)

        src_eq    = source_ids.unsqueeze(0) == source_ids.unsqueeze(1)
        diag_mask = torch.eye(B, dtype=torch.bool, device=z.device)
        pos_mask  = src_eq & ~diag_mask
        if not pos_mask.any():
            return torch.tensor(0.0, device=z.device)

        sim = torch.mm(z_norm, z_norm.T) / self.tau
        sim = sim.masked_fill(diag_mask, float("-inf"))

        Q = self.queue_filled
        if Q > 0 and self.use_hard_negatives and self.hnr > 0:
            q_emb  = F.normalize(self.queue_emb[:Q], dim=-1, eps=self.eps)
            q_src  = self.queue_src[:Q]
            sc     = torch.mm(z_norm, q_emb.T) / self.tau
            sc     = sc.masked_fill(source_ids.unsqueeze(1) == q_src.unsqueeze(0), float("-inf"))
            hk     = min(int(B * self.hnr), Q)
            if hk > 0:
                top_vals, _ = sc.topk(hk, dim=1)
                sim_full = torch.cat([sim, top_vals], dim=1)
            else:
                sim_full = sim
        else:
            sim_full = sim

        total, n_pairs = torch.tensor(0.0, device=z.device), 0
        for i in range(B):
            pos_idx = pos_mask[i].nonzero(as_tuple=False).squeeze(1)
            if len(pos_idx) == 0:
                continue
            valid = ~torch.isinf(sim_full[i])
            if not valid.any():
                continue
            denom  = torch.logsumexp(sim_full[i][valid], dim=0)
            loss_i = -sim_full[i, pos_idx].mean() + denom
            if not torch.isnan(loss_i):
                total  = total + loss_i
                n_pairs += 1
        return total / max(n_pairs, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 12.  Learnable Rarity Scorer (with ablation support)
# ─────────────────────────────────────────────────────────────────────────────

class LearnableRarityScorer(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int, alpha: float = 0.05,
                 disabled: bool = False):
        super().__init__()
        self.disabled = disabled
        self.alpha = alpha
        if not disabled:
            self.register_buffer("norm_mean",  torch.zeros(num_nodes, hidden_dim))
            self.register_buffer("norm_var",   torch.ones(num_nodes,  hidden_dim))
            self.register_buffer("n_updates",  torch.zeros(num_nodes, dtype=torch.long))
            self.score_mlp = nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(),
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1), nn.Sigmoid(),
            )
        self.eps = 1e-8

    @torch.no_grad()
    def update(self, node_ids: torch.Tensor, embeddings: torch.Tensor):
        if self.disabled:
            return
        embeddings = torch.nan_to_num(embeddings, nan=0.0)
        for i, nid in enumerate(node_ids):
            emb = embeddings[i].detach()
            if self.n_updates[nid] == 0:
                self.norm_mean[nid] = emb
                self.norm_var[nid]  = torch.ones_like(emb)
            else:
                self.norm_mean[nid] = (1 - self.alpha) * self.norm_mean[nid] + self.alpha * emb
                self.norm_var[nid]  = ((1 - self.alpha) * self.norm_var[nid]
                                       + self.alpha * (emb - self.norm_mean[nid]) ** 2)
            self.n_updates[nid] += 1

    def _mahal(self, node_ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            return torch.zeros(len(node_ids), 1, device=embeddings.device)
        scores = torch.zeros(len(node_ids), 1, device=embeddings.device)
        for i, nid in enumerate(node_ids):
            if self.n_updates[nid] < 2:
                continue
            diff      = embeddings[i] - self.norm_mean[nid]
            var       = self.norm_var[nid] + self.eps
            scores[i] = torch.mean(diff ** 2 / var)
        return scores

    def forward(self, node_ids: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        if self.disabled:
            return torch.zeros(len(node_ids), device=embeddings.device)
        embeddings = torch.nan_to_num(embeddings, nan=0.0)
        mahal = self._mahal(node_ids, embeddings)
        out   = self.score_mlp(torch.cat([embeddings, mahal], -1)).squeeze(-1)
        return torch.nan_to_num(out, nan=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 13.  Reconstruction & Temporal Consistency
# ─────────────────────────────────────────────────────────────────────────────

class EdgeReconHead(nn.Module):
    def __init__(self, hidden_dim: int, edge_feat_dim: int):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, edge_feat_dim),
        )
    def forward(self, h_src, h_dst):
        return self.dec(torch.cat([h_src, h_dst], -1))


class TemporalConsistencyLoss(nn.Module):
    def forward(self, h_prev, h_curr, mask):
        return ((h_curr[mask] - h_prev[mask]) ** 2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 14.  MITRE ATT&CK Explainer
# ─────────────────────────────────────────────────────────────────────────────

class MITREExplainer:
    _PORT_MAP: Dict[int, List[str]] = {
        22:   ["T1021.004", "T1046"], 3389: ["T1021.001", "T1110.003"],
        445:  ["T1021.002", "T1550.002"], 80: ["T1190", "T1071.001"],
        443:  ["T1190", "T1071.001"],  1433: ["T1508", "T1213"],
        3306: ["T1508", "T1213"],      5900: ["T1021.005"],
        53:   ["T1046", "T1568"],        25: ["T1048", "T1566"],
        21:   ["T1071.002", "T1046"],
    }
    _KEYWORD_MAP: Dict[str, List[str]] = {
        "scan":          ["T1046", "T1595"],
        "bruteforce":    ["T1110", "T1110.001"],
        "ftp-bruteforce":["T1110.001", "T1071.002"],
        "ssh-bruteforce":["T1110.004", "T1021.004"],
        "infiltration":  ["T1078", "T1190"],
        "rdp":           ["T1021.001"],
        "smb":           ["T1021.002"],
        "ssh":           ["T1021.004"],
        "dns":           ["T1568", "T1046"],
    }

    @classmethod
    def tag(cls, chain: str, ports: List[int], labels: List[str] = None) -> List[str]:
        from typing import Set as _Set
        techniques: _Set[str] = set()
        chain_lower = chain.lower()
        for kw, techs in cls._KEYWORD_MAP.items():
            if kw in chain_lower:
                techniques.update(techs)
        for port in ports:
            techniques.update(cls._PORT_MAP.get(port, []))
        if labels:
            for lbl in labels:
                for kw, techs in cls._KEYWORD_MAP.items():
                    if kw in lbl.lower():
                        techniques.update(techs)
        return sorted(techniques)


# ─────────────────────────────────────────────────────────────────────────────
# 15.  Lateral Movement Scorer
# ─────────────────────────────────────────────────────────────────────────────

class LateralMovementScorer:
    def __init__(self, id2ip: Dict[int, str], cfg: Config):
        self.id2ip = id2ip
        self.cfg   = cfg

    def compute_path_scores(self, walk_dicts, walk_embs, edge_scores_map,
                            recon_err_map=None, rarity_map=None,
                            path_logits_map=None, node_emb_map=None,
                            label_map=None) -> List[Dict]:
        results = []
        emb_norms = walk_embs.norm(dim=-1).cpu().numpy()
        for i, wd in enumerate(walk_dicts):
            if i >= walk_embs.size(0):
                break
            ids = [w[0] for w in wd["walk"]]
            recon_score  = float(np.mean(
                [float(np.mean(recon_err_map.get((u, v), [0.0])))
                 for u, v in zip(ids[:-1], ids[1:])])) if recon_err_map else 0.0
            rarity_score = float(np.mean([rarity_map.get(n, 0.0) for n in ids])) if rarity_map else 0.0
            path_nlp     = float(path_logits_map.get(i, 0.0)) if path_logits_map else 0.0
            path_score   = (self.cfg.path_score_w_recon  * recon_score
                           + self.cfg.path_score_w_rarity * rarity_score
                           + self.cfg.path_score_w_likelihood * path_nlp)
            ports = []
            node_labels = []
            for nid in ids:
                ip_str = self.id2ip.get(nid, str(nid))
                if "svc:" in ip_str:
                    try:
                        p = int(ip_str.split(":")[1].split("/")[0])
                        ports.append(p)
                    except Exception:
                        pass
                if label_map and nid in label_map:
                    node_labels.append(label_map[nid])
            chain_str  = " -> ".join(self.id2ip.get(n, str(n)) for n in ids)
            techniques = MITREExplainer.tag(chain_str, ports, node_labels)
            results.append({
                "chain": chain_str, "node_ids": ids, "path_score": path_score,
                "recon_score": recon_score, "rarity_score": rarity_score,
                "path_neg_logprob": path_nlp,
                "mitre_techniques": techniques,
                "risk_level": ("HIGH"   if path_score > 0.7
                               else "MEDIUM" if path_score > 0.4 else "LOW"),
            })
        return sorted(results, key=lambda x: -x["path_score"])

    def compute_host_scores(self, path_records, num_nodes):
        scores = np.zeros(num_nodes)
        for rec in path_records:
            src = rec["node_ids"][0]
            scores[src] = max(scores[src], rec["path_score"])
        return scores

    def top_paths(self, records, k=10): return records[:k]
    def top_hosts(self, scores, k=10):
        top = np.argsort(scores)[::-1][:k]
        return [(self.id2ip.get(i, str(i)), float(scores[i])) for i in top]


# ─────────────────────────────────────────────────────────────────────────────
# 16.  Full THGNN  Model with Ablation Support
# ─────────────────────────────────────────────────────────────────────────────

class THGNNv2_3_CIC(nn.Module):
    """
    THGNN full architecture wired to CIC-IDS2018 features.
    All components identical to the AIT-NDS v2.3 except:
      • CategoricalEmbedder has no crypto/isint flag embeddings.
      • _embed_edges reads CIC column names.
    """
    def __init__(self, cfg: Config, num_nodes: int, num_proto: int,
                 ablation_flags: Dict):
        super().__init__()
        self.cfg = cfg
        self.ablation = ablation_flags
        H = cfg.hidden_dim

        # Apply ablation flags
        use_edge_types = ablation_flags.get("use_edge_types", True)
        use_memory = ablation_flags.get("use_memory", True)
        use_rarity = ablation_flags.get("use_rarity", True)
        use_hard_negatives = ablation_flags.get("use_hard_negatives", True)
        walk_encoder_type = ablation_flags.get("walk_encoder", cfg.walk_encoder)
        num_hec_layers = ablation_flags.get("num_hec_layers", cfg.num_hec_layers)

        self.cat_emb = CategoricalEmbedder(cfg, num_proto, use_edge_types=use_edge_types)
        edge_dim_total = cfg.edge_feat_dim + self.cat_emb.out_dim

        node_in = cfg.node_feat_dim + (cfg.memory_dim if use_memory else 0)
        self.node_proj = nn.Linear(node_in, H)

        self.hec_layers = nn.ModuleList([
            HeterogeneousEdgeConv(H, edge_dim_total, H, cfg.num_heads,
                                  cfg.num_edge_types, cfg.dropout,
                                  use_edge_types=use_edge_types)
            for _ in range(num_hec_layers)
        ])
        
        self.memory = TemporalNodeMemory(num_nodes, cfg.memory_dim,
                                          edge_dim_total,
                                          cfg.memory_decay_half_life,
                                          disabled=not use_memory)
        self.walk_encoder = WalkEncoder(H, walk_encoder_type,
                                         cfg.walk_encoder_layers,
                                         cfg.walk_encoder_heads, cfg.dropout)
        self.recon_head   = EdgeReconHead(H, cfg.edge_feat_dim)
        self.tc_loss_fn   = TemporalConsistencyLoss()
        self.infonce      = HardNegativeInfoNCE(cfg.temperature, cfg.hard_neg_ratio,
                                                cfg.hard_neg_queue_size, H,
                                                use_hard_negatives=use_hard_negatives)
        self.path_recon_head = PathReconstructionHead(H, num_nodes)
        self.rarity_scorer   = LearnableRarityScorer(num_nodes, H, disabled=not use_rarity)
        self.edge_scorer     = nn.Sequential(
            nn.Linear(H * 2 + edge_dim_total, H), nn.GELU(), nn.Linear(H, 1)
        )
        self.eps = 1e-8

    # ── helpers ──────────────────────────────────────────────────────────────

    def _embed_edges(self, df_snap: pd.DataFrame,
                     numeric_ef: torch.Tensor) -> Tuple:
        src_ids    = torch.tensor(df_snap["src_id"].values,      dtype=torch.long)
        dst_ids    = torch.tensor(df_snap["dst_id"].values,      dtype=torch.long)
        edge_index = torch.stack([src_ids, dst_ids], 0)

        proto_ids  = torch.tensor(df_snap["protocol_id"].values.astype(int),  dtype=torch.long)
        src_ports  = torch.tensor(df_snap["src_port"].values.astype(int),     dtype=torch.long)
        dst_ports  = torch.tensor(df_snap["dst_port"].values.astype(int),     dtype=torch.long)
        dst_pcats  = torch.tensor(df_snap["dst_port_cat"].values.astype(int), dtype=torch.long)
        edge_types = torch.tensor(df_snap["edge_type"].values.astype(int),    dtype=torch.long)

        cat_feats = self.cat_emb(proto_ids, src_ports, dst_ports, dst_pcats, edge_types)
        edge_feat = torch.cat([numeric_ef, cat_feats], -1)

        ts       = df_snap["ts_unix"].values
        ts_norm  = (ts - ts.min()) / (ts.max() - ts.min() + self.eps)
        delta_t  = torch.tensor(
            np.abs(np.diff(ts_norm, prepend=ts_norm[:1])).astype(np.float32))
        return edge_index, edge_feat, edge_types, delta_t

    def _encode_walks(self, walk_dicts, h, current_ts):
        device = h.device
        L      = self.cfg.walk_length + 1
        emb_list, ts_list, src_list, step_list = [], [], [], []
        for wd in walk_dicts:
            walk = wd["walk"]
            if len(walk) < L:
                continue
            walk = walk[:L]
            ids  = [w[0] for w in walk]
            ts   = np.array([w[1] for w in walk], dtype=np.float32)
            ts_n = (ts - ts.min()) / (ts.max() - ts.min() + self.eps)
            step_h = h[torch.tensor(ids, device=device)]
            emb_list.append(step_h)
            ts_list.append(torch.tensor(ts_n, device=device))
            src_list.append(wd["source"])
            step_list.append(step_h)
        if not emb_list:
            return (torch.zeros(0, self.cfg.hidden_dim, device=device),
                    torch.zeros(0, dtype=torch.long,   device=device),
                    None)
        walk_embs  = self.walk_encoder(torch.stack(emb_list), torch.stack(ts_list))
        source_ids = torch.tensor(src_list, dtype=torch.long, device=device)
        step_embs  = torch.stack(step_list) if step_list else None
        return walk_embs, source_ids, step_embs

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, node_feats_static, df_snap, numeric_ef,
                walk_dicts, current_ts, h_prev):
        node_feats_static = torch.nan_to_num(node_feats_static, nan=0.0)
        numeric_ef        = torch.nan_to_num(numeric_ef,        nan=0.0)

        N       = node_feats_static.size(0)
        all_ids = torch.arange(N, device=node_feats_static.device)
        
        # Apply memory ablation
        if self.ablation.get("use_memory", True):
            mem = self.memory.get(all_ids, current_ts)
            h_input = torch.cat([node_feats_static, mem], -1)
        else:
            h_input = node_feats_static
            
        h = torch.nan_to_num(self.node_proj(h_input), nan=0.0)

        edge_index, edge_feat, edge_type, delta_t = self._embed_edges(df_snap, numeric_ef)
        edge_index = edge_index.to(h.device)
        edge_feat  = edge_feat.to(h.device)
        edge_type  = edge_type.to(h.device)
        delta_t    = delta_t.to(h.device)

        for hec in self.hec_layers:
            h = hec(h, edge_index, edge_feat, edge_type, delta_t)

        h_src = h[edge_index[0]]
        h_dst = h[edge_index[1]]

        # ── Reconstruction loss (always enabled) ──────────────────────────────
        recon_feat = self.recon_head(h_src, h_dst)
        recon_loss = F.mse_loss(recon_feat, numeric_ef.to(h.device))
        if torch.isnan(recon_loss):
            recon_loss = torch.tensor(1.0, device=h.device)

        with torch.no_grad():
            per_edge_recon_err = F.mse_loss(
                recon_feat, numeric_ef.to(h.device), reduction="none").mean(-1)
            per_edge_recon_err = torch.nan_to_num(per_edge_recon_err, nan=0.0)

        # ── Contrastive + path recon (ablatable) ────────────────────────────────
        infonce_loss = torch.tensor(0.0, device=h.device)
        path_recon_loss = torch.tensor(0.0, device=h.device)
        walk_embs = None

        if self.ablation.get("use_contrastive", True) and walk_dicts and len(walk_dicts) >= 2:
            walk_embs, source_ids, step_embs = self._encode_walks(walk_dicts, h, current_ts)
            if walk_embs is not None and walk_embs.size(0) >= 2:
                infonce_loss = self.infonce(walk_embs, source_ids)
                if torch.isnan(infonce_loss):
                    infonce_loss = torch.tensor(0.0, device=h.device)
                self.infonce.enqueue(walk_embs.detach(), source_ids.detach())

                if self.ablation.get("use_path_recon", True) and step_embs is not None and step_embs.size(1) > 1:
                    logits = self.path_recon_head(step_embs)
                    target_ids = []
                    for wd in walk_dicts[:step_embs.size(0)]:
                        walk = wd["walk"][:self.cfg.walk_length + 1]
                        target_ids.extend([w[0] for w in walk[1:]])
                    if target_ids:
                        targets = torch.tensor(
                            target_ids[:logits.size(0) * self.cfg.walk_length],
                            dtype=torch.long, device=h.device)
                        logits_flat = logits.reshape(-1, logits.size(-1))
                        n = min(logits_flat.size(0), targets.size(0))
                        if n > 1:
                            path_recon_loss = F.cross_entropy(logits_flat[:n], targets[:n])
                            if torch.isnan(path_recon_loss):
                                path_recon_loss = torch.tensor(0.0, device=h.device)

        # ── Temporal consistency (ablatable) ────────────────────────────────────
        tc_loss = torch.tensor(0.0, device=h.device)
        if self.ablation.get("use_temporal", True) and h_prev is not None:
            tc_loss = self.tc_loss_fn(h_prev.to(h.device), h,
                                      torch.ones(N, dtype=torch.bool, device=h.device))
            if torch.isnan(tc_loss):
                tc_loss = torch.tensor(0.0, device=h.device)

        # ── Rarity loss (ablatable) ─────────────────────────────────────────────
        src_ids_tensor = torch.tensor(df_snap["src_id"].values,
                                       dtype=torch.long, device=h.device)
        rarity_scores = self.rarity_scorer(src_ids_tensor, h_src)
        rarity_loss = rarity_scores.mean() if self.ablation.get("use_rarity", True) else torch.tensor(0.0, device=h.device)
        if torch.isnan(rarity_loss):
            rarity_loss = torch.tensor(0.0, device=h.device)

        # ── Total loss (ablation-aware) ──────────────────────────────────────────
        loss = (self.cfg.lambda_recon * recon_loss
              + (self.cfg.lambda_contrastive if self.ablation.get("use_contrastive", True) else 0.0) * infonce_loss
              + (self.cfg.lambda_temporal if self.ablation.get("use_temporal", True) else 0.0) * tc_loss
              + (self.cfg.lambda_path_recon if self.ablation.get("use_path_recon", True) else 0.0) * path_recon_loss
              + (self.cfg.lambda_rarity if self.ablation.get("use_rarity", True) else 0.0) * rarity_loss)
        if torch.isnan(loss):
            loss = recon_loss

        edge_scores = torch.nan_to_num(
            self.edge_scorer(torch.cat([h_src, h_dst, edge_feat], -1)).squeeze(-1).detach(),
            nan=0.0)

        if self.ablation.get("use_memory", True):
            self.memory.update(edge_index[0], edge_index[1], edge_feat.detach(), current_ts)
        if self.ablation.get("use_rarity", True):
            self.rarity_scorer.update(src_ids_tensor, h_src.detach())

        return {
            "loss": loss, "recon_loss": recon_loss, "infonce_loss": infonce_loss,
            "tc_loss": tc_loss, "path_recon_loss": path_recon_loss,
            "rarity_loss": rarity_loss,
            "h": h.detach(), "edge_scores": edge_scores, "walk_embs": walk_embs,
            "per_edge_recon_err": per_edge_recon_err, "rarity_scores": rarity_scores.detach(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 17.  Trainer with Ablation Support
# ─────────────────────────────────────────────────────────────────────────────

class THGNNv2Trainer:
    def __init__(self, cfg: Config, model: THGNNv2_3_CIC, meta: Dict):
        self.cfg   = cfg
        self.model = model.to(DEVICE)
        self.meta  = meta
        self.opt   = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = CosineAnnealingLR(self.opt, T_max=max(cfg.epochs, 1))
        self.walk_gen = MemoryAwareWalkGenerator(cfg.walk_length, cfg.num_walks,
                                                  cfg.walk_temporal_window_s,
                                                  cfg.walk_exploration_epsilon)
        self.scaler   = GradScaler(enabled=cfg.mixed_precision and DEVICE.type == "cuda")
        self.threshold = 0.5

    def _numeric_ef(self, snap: pd.DataFrame) -> torch.Tensor:
        arr = snap[_CIC_NUMERIC_COLS].values.astype(np.float32)
        if arr.shape[1] < self.cfg.edge_feat_dim:
            arr = np.pad(arr, ((0, 0), (0, self.cfg.edge_feat_dim - arr.shape[1])))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tensor(arr[:, :self.cfg.edge_feat_dim], dtype=torch.float32)

    def _split_snapshots(self, df: pd.DataFrame):
        t0, t1 = df["ts_unix"].min(), df["ts_unix"].max()
        w      = self.cfg.temporal_window_s
        snaps  = []
        t = t0
        while t < t1:
            s = df[(df["ts_unix"] >= t) & (df["ts_unix"] < t + w)]
            if len(s) > 0:
                snaps.append(s)
            t += w
        if not snaps:
            snaps = [df]
        n_tr = max(int(len(snaps) * self.cfg.train_ratio), 1)
        n_va = int(len(snaps) * self.cfg.val_ratio)
        return (snaps[:n_tr],
                snaps[n_tr:n_tr + n_va] or snaps[-1:],
                snaps[n_tr + n_va:]     or snaps[-1:])

    def train(self, df: pd.DataFrame, node_feats: torch.Tensor) -> Dict:
        tr, va, te = self._split_snapshots(df)
        adj = self.walk_gen.build_adjacency(
            df[df["ts_unix"] < df["ts_unix"].quantile(self.cfg.train_ratio)])

        best_val_loss = float("inf")
        patience_ctr  = 0
        best_state    = None
        val_sc_all, val_lb_all = [], []

        ablation_name = self.cfg.ablation_config
        ablation_info = ABLATION_CONFIGS.get(ablation_name, ABLATION_CONFIGS["full"])
        
        logger.info("Snapshots — Train:%d  Val:%d  Test:%d", len(tr), len(va), len(te))
        print(f"\n{'='*65}\n  THGNN v2.3-CIC — {ablation_name.upper()} ({self.cfg.epochs} epochs)\n  {ablation_info['description']}\n{'='*65}\n",
              flush=True)

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            self.model.memory.reset()
            tot_loss = recon_t = cont_t = tc_t = path_t = rar_t = 0.0
            h_prev, n_valid = None, 0

            for i, snap in enumerate(tr):
                if len(snap) == 0:
                    continue
                nef    = self._numeric_ef(snap).to(DEVICE)
                cur_ts = float(snap["ts_unix"].mean())
                nfw    = build_node_features_window(snap, self.meta["num_nodes"]).to(DEVICE)
                wds    = self.walk_gen.generate_walks(
                            self.walk_gen.build_adjacency(snap), self.model.memory, cur_ts)

                self.opt.zero_grad(set_to_none=True)
                try:
                    with autocast(enabled=self.cfg.mixed_precision and DEVICE.type == "cuda"):
                        out  = self.model(nfw, snap, nef, wds, cur_ts, h_prev)
                        loss = out["loss"]
                    if torch.isnan(loss):
                        continue
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.opt)
                    self.scaler.update()

                    tot_loss += loss.item()
                    recon_t  += out["recon_loss"].item()
                    cont_t   += out["infonce_loss"].item()
                    tc_t     += out["tc_loss"].item()
                    path_t   += out["path_recon_loss"].item()
                    rar_t    += out["rarity_loss"].item()
                    h_prev    = out["h"]
                    n_valid  += 1
                except Exception:
                    continue

                if (i + 1) % max(1, len(tr) // 10) == 0 and n_valid > 0:
                    print(f"  Epoch {epoch:02d} | {(i+1)/len(tr)*100:.0f}%"
                          f" | loss={tot_loss/n_valid:.4f}", flush=True)

            if n_valid == 0:
                continue

            tot_loss /= n_valid
            recon_t /= n_valid
            cont_t /= n_valid
            tc_t /= n_valid
            path_t /= n_valid
            rar_t /= n_valid
            self.sched.step()

            val_loss, val_sc, val_lb = self._eval_pass(va)
            if not np.isnan(val_loss):
                val_sc_all, val_lb_all = val_sc, val_lb

            print(f"\n✓ Epoch {epoch:02d}: train={tot_loss:.4f}"
                  f" (rec={recon_t:.4f} con={cont_t:.4f}"
                  f" tc={tc_t:.4f} path={path_t:.4f} rar={rar_t:.4f})")
            print(f"            val={val_loss:.4f}")
            print("-" * 65, flush=True)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss, patience_ctr = val_loss, 0
                best_state = copy.deepcopy(self.model.state_dict())
                print(f"  → Best saved (val_loss={best_val_loss:.4f})", flush=True)
            else:
                patience_ctr += 1
                if patience_ctr >= self.cfg.patience:
                    print("\n  → Early stopping.\n", flush=True)
                    break

        if best_state:
            self.model.load_state_dict(best_state)
        
        if len(val_sc_all) > 0:
            self._optimise_threshold(np.array(val_sc_all), np.array(val_lb_all))

        print(f"\n{'='*65}\nFinal evaluation on test set...\n", flush=True)
        return self._evaluate(te, adj)

    @torch.no_grad()
    def _eval_pass(self, snaps) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        total, scores, labels, h_prev, n_valid = 0.0, [], [], None, 0
        for snap in snaps:
            if len(snap) == 0:
                continue
            try:
                nef    = self._numeric_ef(snap).to(DEVICE)
                cur_ts = float(snap["ts_unix"].mean())
                nfw    = build_node_features_window(snap, self.meta["num_nodes"]).to(DEVICE)
                wds    = self.walk_gen.generate_walks(self.walk_gen.build_adjacency(snap))
                out    = self.model(nfw, snap, nef, wds, cur_ts, h_prev)
                lv     = out["loss"].item()
                if not np.isnan(lv):
                    total  += lv
                    n_valid+= 1
                h_prev = out["h"]
                scores.append(out["edge_scores"].cpu().numpy())
                labels.append(snap["is_attack"].values)
            except Exception:
                continue
        sc = np.concatenate(scores) if scores else np.array([])
        lb = np.concatenate(labels) if labels else np.array([])
        return total / max(n_valid, 1), sc, lb

    def _optimise_threshold(self, val_scores: np.ndarray, val_labels: np.ndarray):
        mask = ~np.isnan(val_scores)
        vs, vl = val_scores[mask], val_labels[mask]
        if len(vs) < 2 or len(np.unique(vl)) < 2:
            self.threshold = 0.5
            return
        sc_min, sc_max = vs.min(), vs.max()
        if sc_max - sc_min < 1e-8:
            self.threshold = 0.5
            return
        vs_n = (vs - sc_min) / (sc_max - sc_min + 1e-8)
        fpr, tpr, ths = roc_curve(vl, vs_n)
        best = int(np.argmax(tpr - fpr))
        self.threshold = float(ths[best]) if best < len(ths) else 0.5
        logger.info("Youden-J threshold: %.4f", self.threshold)

    @torch.no_grad()
    def _evaluate(self, test_snaps, train_adj) -> Dict:
        self.model.eval()
        self.model.memory.reset()

        all_scores, all_labels  = [], []
        edge_scores_map         = defaultdict(list)
        recon_err_map           = defaultdict(list)
        rarity_map: Dict        = {}
        label_map:  Dict        = {}
        all_walk_dicts          = []
        all_walk_embs           = []
        h_prev                  = None

        for i, snap in enumerate(test_snaps):
            if len(snap) == 0:
                continue
            try:
                nef    = self._numeric_ef(snap).to(DEVICE)
                cur_ts = float(snap["ts_unix"].mean())
                nfw    = build_node_features_window(snap, self.meta["num_nodes"]).to(DEVICE)
                wds    = self.walk_gen.generate_walks(
                            self.walk_gen.build_adjacency(snap), self.model.memory, cur_ts)
                out    = self.model(nfw, snap, nef, wds, cur_ts, h_prev)
                h_prev = out["h"]

                scores = out["edge_scores"].cpu().numpy()
                labels = snap["is_attack"].values
                valid  = ~np.isnan(scores)
                if not valid.any():
                    continue
                all_scores.append(scores[valid])
                all_labels.append(labels[valid])

                for j, (s, d) in enumerate(zip(snap["src_id"].values, snap["dst_id"].values)):
                    if j < len(scores) and not np.isnan(scores[j]):
                        edge_scores_map[(int(s), int(d))].append(float(scores[j]))
                        recon_err_map[(int(s), int(d))].append(
                            float(out["per_edge_recon_err"][j].item()))
                for j, sid in enumerate(snap["src_id"].values):
                    if j < len(out["rarity_scores"]):
                        rarity_map[int(sid)] = float(out["rarity_scores"][j].item())
                for idx, nid in enumerate(snap["src_id"].values):
                    if idx < len(snap["label"].values):
                        label_map[int(nid)] = snap["label"].values[idx]

                if out["walk_embs"] is not None and out["walk_embs"].size(0) > 0:
                    all_walk_dicts.extend(wds[:out["walk_embs"].size(0)])
                    all_walk_embs.append(out["walk_embs"].cpu())

                if (i + 1) % max(1, len(test_snaps) // 5) == 0:
                    print(f"  Test eval {(i+1)/len(test_snaps)*100:.0f}%", flush=True)
            except Exception:
                continue

        if not all_scores:
            return {"auroc": float("nan"), "ap": float("nan"), "f1": 0.0,
                    "precision": 0.0, "recall": 0.0}

        sc_arr = np.concatenate(all_scores)
        lb_arr = np.concatenate(all_labels)
        sc_min, sc_max = sc_arr.min(), sc_arr.max()
        sc_norm  = (sc_arr - sc_min) / (sc_max - sc_min + 1e-8) if sc_max > sc_min else np.zeros_like(sc_arr)
        preds    = (sc_norm >= self.threshold).astype(int)

        if lb_arr.sum() > 0 and (len(lb_arr) - lb_arr.sum()) > 0:
            auroc = roc_auc_score(lb_arr, sc_norm)
            ap    = average_precision_score(lb_arr, sc_norm)
        else:
            auroc = ap = float("nan")

        f1   = f1_score(lb_arr, preds, zero_division=0)
        prec = precision_score(lb_arr, preds, zero_division=0)
        rec  = recall_score(lb_arr, preds, zero_division=0)
        cm   = confusion_matrix(lb_arr, preds)

        logger.info("=" * 60)
        logger.info("EDGE-LEVEL EVALUATION (CIC-IDS2018, proxy labels)")
        logger.info("  AUROC  : %.4f", auroc if not np.isnan(auroc) else 0.0)
        logger.info("  AP     : %.4f", ap    if not np.isnan(ap)    else 0.0)
        logger.info("  F1     : %.4f", f1)
        logger.info("  Prec   : %.4f", prec)
        logger.info("  Recall : %.4f", rec)
        logger.info("Confusion Matrix:\n%s", cm)
        logger.info("=" * 60)

        scorer      = LateralMovementScorer(self.meta["id2ip"], self.cfg)
        path_records= []
        if all_walk_embs:
            wec = torch.cat(all_walk_embs, 0)
            if wec.size(0) > 0:
                path_records = scorer.compute_path_scores(
                    all_walk_dicts, wec, edge_scores_map,
                    recon_err_map, rarity_map, None, None, label_map)
        host_scores = scorer.compute_host_scores(path_records, self.meta["num_nodes"])
        top_p       = scorer.top_paths(path_records)
        top_h       = scorer.top_hosts(host_scores)

        logger.info("Top-10 Suspicious Lateral Movement Chains:")
        for r, rec in enumerate(top_p, 1):
            logger.info("  %2d. [score=%.4f | recon=%.4f | rarity=%.4f | risk=%s | MITRE=%s] %s",
                        r, rec["path_score"], rec["recon_score"], rec["rarity_score"],
                        rec["risk_level"], rec["mitre_techniques"][:3], rec["chain"][:150])
        logger.info("Top-10 Suspicious Nodes:")
        for r, (node, sc) in enumerate(top_h, 1):
            logger.info("  %2d. [score=%.4f] %s", r, sc, node)

        return {"auroc": auroc, "ap": ap, "f1": f1, "precision": prec, "recall": rec,
                "threshold": self.threshold, "confusion_matrix": cm.tolist(),
                "top_paths": top_p, "top_hosts": top_h, "host_scores": host_scores,
                "ablation_config": self.cfg.ablation_config}


# ─────────────────────────────────────────────────────────────────────────────
# 18.  Ablation Study Runner
# ─────────────────────────────────────────────────────────────────────────────

class AblationStudyRunner:
    def __init__(self, base_cfg: Config, df: pd.DataFrame, meta: Dict):
        self.base_cfg = base_cfg
        self.df = df
        self.meta = meta
        self.results = {}
        
    def run_single_ablation(self, ablation_name: str, seed: int) -> Dict:
        """Run a single ablation configuration with a specific seed"""
        set_seed(seed)
        
        # Get ablation flags
        ablation_info = ABLATION_CONFIGS[ablation_name]
        ablation_flags = ablation_info["flags"]
        
        # Create config copy with ablation setting
        cfg = copy.deepcopy(self.base_cfg)
        cfg.ablation_config = ablation_name
        cfg.num_proto = self.meta["num_proto"]
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Running ablation: {ablation_info['name']} (seed={seed})")
        logger.info(f"Description: {ablation_info['description']}")
        logger.info(f"{'='*70}")
        
        # Create model with ablation flags
        model = THGNNv2_3_CIC(cfg, self.meta["num_nodes"], self.meta["num_proto"], ablation_flags)
        
        # Create trainer
        trainer = THGNNv2Trainer(cfg, model, self.meta)
        
        # Train and evaluate
        metrics = trainer.train(self.df, None)
        metrics["ablation_name"] = ablation_name
        metrics["ablation_display_name"] = ablation_info["name"]
        metrics["seed"] = seed
        metrics["flags"] = ablation_flags
        
        return metrics
    
    def run_full_ablation_study(self) -> Dict:
        """Run all ablation configurations with multiple seeds"""
        os.makedirs(self.base_cfg.ablation_output_dir, exist_ok=True)
        
        all_results = {}
        
        for ablation_name in ABLATION_CONFIGS.keys():
            seed_results = []
            
            for seed in range(self.base_cfg.ablation_seeds):
                try:
                    metrics = self.run_single_ablation(ablation_name, seed)
                    seed_results.append(metrics)
                    
                    # Save individual result
                    result_path = os.path.join(
                        self.base_cfg.ablation_output_dir,
                        f"{ablation_name}_seed{seed}.json"
                    )
                    # Convert numpy types to Python types for JSON
                    serializable_metrics = {}
                    for k, v in metrics.items():
                        if isinstance(v, np.floating):
                            serializable_metrics[k] = float(v)
                        elif isinstance(v, np.integer):
                            serializable_metrics[k] = int(v)
                        elif isinstance(v, np.ndarray):
                            serializable_metrics[k] = v.tolist()
                        else:
                            serializable_metrics[k] = v
                    with open(result_path, "w") as f:
                        json.dump(serializable_metrics, f, indent=2)
                        
                except Exception as e:
                    logger.error(f"Failed {ablation_name} seed {seed}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if seed_results:
                # Compute statistics across seeds
                all_results[ablation_name] = {
                    "individual": seed_results,
                    "stats": self._compute_statistics(seed_results),
                    "display_name": ABLATION_CONFIGS[ablation_name]["name"],
                    "description": ABLATION_CONFIGS[ablation_name]["description"],
                }
        
        # Save summary
        self._save_summary(all_results)
        self._print_summary_table(all_results)
        
        return all_results
    
    def _compute_statistics(self, results: List[Dict]) -> Dict:
        """Compute mean and std across seeds for key metrics"""
        stats = {}
        metrics_keys = ["auroc", "ap", "f1", "precision", "recall"]
        
        for key in metrics_keys:
            values = [r.get(key, float("nan")) for r in results if not np.isnan(r.get(key, float("nan")))]
            if values:
                stats[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
            else:
                stats[key] = {"mean": float("nan"), "std": float("nan")}
        
        return stats
    
    def _save_summary(self, all_results: Dict):
        """Save ablation study summary to file"""
        summary_path = os.path.join(self.base_cfg.ablation_output_dir, "ablation_summary.json")
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "base_config": {
                "hidden_dim": self.base_cfg.hidden_dim,
                "epochs": self.base_cfg.epochs,
                "lr": self.base_cfg.lr,
                "seeds_per_config": self.base_cfg.ablation_seeds,
            },
            "results": {}
        }
        
        for ablation_name, data in all_results.items():
            summary["results"][ablation_name] = {
                "display_name": data["display_name"],
                "description": data["description"],
                "stats": data["stats"],
            }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Ablation summary saved to {summary_path}")
    
    def _print_summary_table(self, all_results: Dict):
        """Print formatted summary table of ablation results"""
        print("\n" + "=" * 120)
        print("ABLATION STUDY SUMMARY - CIC-IDS2018")
        print("=" * 120)
        print(f"{'Configuration':<35} {'AUROC':>15} {'AP':>15} {'F1':>15} {'Precision':>15} {'Recall':>15}")
        print("-" * 120)
        
        # Sort by AUROC (full model first)
        sorted_ablation = sorted(all_results.items(), 
                                key=lambda x: x[1]["stats"].get("auroc", {}).get("mean", 0), 
                                reverse=True)
        
        for ablation_name, data in sorted_ablation:
            stats = data["stats"]
            display_name = data["display_name"][:33]
            
            auroc = stats.get("auroc", {})
            ap = stats.get("ap", {})
            f1 = stats.get("f1", {})
            prec = stats.get("precision", {})
            rec = stats.get("recall", {})
            
            print(f"{display_name:<35} "
                  f"{auroc.get('mean', 0):.4f}±{auroc.get('std', 0):.4f} "
                  f"{ap.get('mean', 0):.4f}±{ap.get('std', 0):.4f} "
                  f"{f1.get('mean', 0):.4f}±{f1.get('std', 0):.4f} "
                  f"{prec.get('mean', 0):.4f}±{prec.get('std', 0):.4f} "
                  f"{rec.get('mean', 0):.4f}±{rec.get('std', 0):.4f}")
        
        print("=" * 120)
        
        # Calculate and print degradation percentages
        print("\n" + "=" * 120)
        print("PERFORMANCE DEGRADATION RELATIVE TO FULL MODEL")
        print("=" * 120)
        
        full_stats = all_results.get("full", {}).get("stats", {})
        full_auroc = full_stats.get("auroc", {}).get("mean", 0)
        
        for ablation_name, data in sorted_ablation:
            if ablation_name == "full":
                continue
            stats = data["stats"]
            auroc = stats.get("auroc", {}).get("mean", 0)
            degradation = ((full_auroc - auroc) / full_auroc) * 100 if full_auroc > 0 else 0
            print(f"{data['display_name']:<35}: AUROC = {auroc:.4f} (Δ = {degradation:+.2f}%)")
        
        print("=" * 120)


# ─────────────────────────────────────────────────────────────────────────────
# 19.  Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, cfg, path):
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, path)
    logger.info("Saved → %s", path)


def main():
    import argparse
    p = argparse.ArgumentParser(description="THGNN v2.3-CIC — FULL ARCHITECTURE WITH ABLATION STUDY")
    p.add_argument("--data",        default="data/cic_ids2018")
    p.add_argument("--max_files",   type=int,   default=None)
    p.add_argument("--sample_frac", type=float, default=None)
    p.add_argument("--max_rows",    type=int,   default=None)
    p.add_argument("--attack_labels",           default=None,
                   help="Comma-separated proxy attack label strings")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--hidden",      type=int,   default=128)
    p.add_argument("--walk_len",    type=int,   default=4)
    p.add_argument("--num_walks",   type=int,   default=20)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--checkpoint",              default="thgnn_v2_3_cic.pt")
    p.add_argument("--seed",        type=int,   default=42)
    
    # Ablation study arguments
    p.add_argument("--ablation", action="store_true", help="Run ablation study")
    p.add_argument("--ablation_config", type=str, default="full", 
                   choices=list(ABLATION_CONFIGS.keys()),
                   help="Single ablation config to run")
    p.add_argument("--ablation_seeds", type=int, default=3,
                   help="Number of seeds for ablation study")
    p.add_argument("--ablation_output_dir", type=str, default="ablation_results_cic",
                   help="Output directory for ablation results")
    
    args = p.parse_args()

    set_seed(args.seed)

    attack_labels = (
        [l.strip() for l in args.attack_labels.split(",") if l.strip()]
        if args.attack_labels else None
    )

    cfg = Config(
        data_path=args.data,
        max_files=args.max_files,
        sample_frac=args.sample_frac,
        max_rows=args.max_rows,
        epochs=args.epochs,
        hidden_dim=args.hidden,
        walk_length=args.walk_len,
        num_walks=args.num_walks,
        lr=args.lr,
        ablation_seeds=args.ablation_seeds,
        ablation_output_dir=args.ablation_output_dir,
        **({"proxy_attack_labels": attack_labels} if attack_labels else {}),
    )

    logger.info("=" * 60)
    logger.info("THGNN v2.3-CIC — FULL ARCHITECTURE WITH ABLATION STUDY")
    logger.info("  data: %s  |  hidden: %d  |  walk_len: %d  |  lr: %.2e",
                cfg.data_path, cfg.hidden_dim, cfg.walk_length, cfg.lr)
    logger.info("  ablation_mode: %s", "ENABLED" if args.ablation else "DISABLED")
    if args.ablation:
        logger.info("  ablation_configs: %s", list(ABLATION_CONFIGS.keys()))
        logger.info("  seeds_per_config: %d", cfg.ablation_seeds)
    logger.info("=" * 60)

    # Load data
    df, meta = load_and_preprocess(cfg)
    cfg.num_proto = meta["num_proto"]

    if args.ablation:
        # Run full ablation study
        runner = AblationStudyRunner(cfg, df, meta)
        results = runner.run_full_ablation_study()
        logger.info("Ablation study completed. Results saved to %s", cfg.ablation_output_dir)
    else:
        # Run single configuration
        if args.ablation_config != "full":
            logger.info(f"Running single ablation: {ABLATION_CONFIGS[args.ablation_config]['name']}")
            ablation_flags = ABLATION_CONFIGS[args.ablation_config]["flags"]
            cfg.ablation_config = args.ablation_config
            model = THGNNv2_3_CIC(cfg, meta["num_nodes"], meta["num_proto"], ablation_flags)
        else:
            logger.info("Running full model")
            model = THGNNv2_3_CIC(cfg, meta["num_nodes"], meta["num_proto"], 
                                 ABLATION_CONFIGS["full"]["flags"])
        
        trainer = THGNNv2Trainer(cfg, model, meta)
        metrics = trainer.train(df, None)
        save_checkpoint(model, cfg, args.checkpoint)

        print("\n" + "=" * 65)
        print(f"  THGNN v2.3-CIC — {cfg.ablation_config.upper()} EVALUATION")
        print("=" * 65)
        for k, label in [("auroc", "AUROC"), ("ap", "Avg Precision"),
                         ("f1", "F1 Score"), ("precision", "Precision"), ("recall", "Recall")]:
            v = metrics.get(k, float("nan"))
            print(f"  {label:<18}: {'NaN' if isinstance(v, float) and math.isnan(v) else f'{v:.4f}'}")
        print("=" * 65)


if __name__ == "__main__":
    main()