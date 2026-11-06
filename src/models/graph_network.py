import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any


class GATLayer(nn.Module):
    """
    Graph Attention Layer implemented in native PyTorch.
    Computes masked multi-head self-attention over node graph neighborhoods.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        leaky_slope: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.leaky_slope = leaky_slope

        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        self.a_src = nn.Parameter(torch.zeros(heads, out_features))
        self.a_dst = nn.Parameter(torch.zeros(heads, out_features))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_nodes = x.size(0)
        # Linear projection: (N, heads * out_features) -> (N, heads, out_features)
        h = self.W(x).view(num_nodes, self.heads, self.out_features)

        src, dst = edge_index[0], edge_index[1]

        # Source and destination node features per edge: (E, heads, out_features)
        h_src = h[src]
        h_dst = h[dst]

        # Edge attention coefficients: (E, heads)
        score_src = (h_src * self.a_src).sum(dim=-1)
        score_dst = (h_dst * self.a_dst).sum(dim=-1)
        edge_scores = F.leaky_relu(score_src + score_dst, negative_slope=self.leaky_slope)

        # Softmax normalization grouped by target node dst
        exp_scores = torch.exp(edge_scores - edge_scores.max())
        
        sum_exp = torch.zeros(num_nodes, self.heads, device=x.device)
        sum_exp.index_add_(0, dst, exp_scores)
        
        alpha = exp_scores / (sum_exp[dst] + 1e-8)
        if self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Aggregate weighted messages to target nodes: (E, heads, out_features)
        weighted_msgs = h_src * alpha.unsqueeze(-1)
        out = torch.zeros(num_nodes, self.heads, self.out_features, device=x.device)
        out.index_add_(0, dst, weighted_msgs)

        if self.concat:
            out = out.view(num_nodes, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)

        att = alpha.mean(dim=-1) if return_attention else None
        return out, att


class InverterGAT(nn.Module):
    """
    2-Layer Graph Attention Network for Fleet Spatial Cross-Validation.
    Predicts inverter power from peer network features with a peer-consensus residual skip connection.
    """
    def __init__(
        self,
        in_features: int = 6,
        hidden_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gat1 = GATLayer(in_features, hidden_dim, heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATLayer(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.out_head = nn.Linear(hidden_dim, 1)

        # Initialize out_head to small values for stable residual learning
        nn.init.normal_(self.out_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.out_head.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # 1. Base peer consensus prediction from peer power (feature 0)
        deg = torch.zeros(num_nodes, 1, device=x.device)
        deg.index_add_(0, dst, torch.ones(edge_index.size(1), 1, device=x.device))
        base_peer_power = torch.zeros(num_nodes, 1, device=x.device)
        base_peer_power.index_add_(0, dst, x[src, 0:1])
        base_peer_power = base_peer_power / (deg + 1e-8)

        # 2. Graph attention refined delta
        h1, att1 = self.gat1(x, edge_index, return_attention=return_attention)
        h1 = F.elu(h1)
        h2, _ = self.gat2(h1, edge_index, return_attention=False)
        h2 = F.elu(h2)
        delta = self.out_head(h2)

        out = F.relu(base_peer_power + delta)
        return out, att1


def build_adjacency_from_correlation(
    df: pd.DataFrame,
    threshold: float = 0.70,
    value_col: str = "AC_POWER",
    inverter_col: str = "INVERTER_ID",
    timestamp_col: str = "DATE_TIME",
) -> Tuple[np.ndarray, torch.Tensor, List[str]]:
    """
    Builds an undirected fleet graph based on pairwise Pearson correlation coefficients.
    """
    pivot = df.pivot(index=timestamp_col, columns=inverter_col, values=value_col)
    inverter_list = sorted(pivot.columns.tolist())
    pivot = pivot[inverter_list]

    corr = pivot.corr().fillna(0.0).to_numpy().copy()
    np.fill_diagonal(corr, 0.0)

    # Thresholding
    adj = (corr >= threshold).astype(float)
    # Ensure graph is symmetric
    adj = np.maximum(adj, adj.T)

    # Guarantee no isolated nodes: connect isolated nodes to highest correlation peer
    for i in range(len(inverter_list)):
        if adj[i].sum() == 0:
            top_peer = int(np.argmax(corr[i]))
            adj[i, top_peer] = 1.0
            adj[top_peer, i] = 1.0

    src, dst = np.where(adj > 0)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return adj, edge_index, inverter_list


def prepare_fleet_snapshots(
    df: pd.DataFrame,
    inverter_list: List[str],
    timestamp_col: str = "DATE_TIME",
    inverter_col: str = "INVERTER_ID",
    power_col: str = "AC_POWER",
) -> List[Dict[str, Any]]:
    """
    Converts temporal fleet records into graph snapshot tensors.
    """
    work_df = df.copy()
    if "PEER_RANK_PCT" not in work_df.columns:
        work_df["PEER_RANK_PCT"] = 0.5
    if "MODEL_RESIDUAL" not in work_df.columns:
        work_df["MODEL_RESIDUAL"] = 0.0
    if "GRID_Z_SCORE" not in work_df.columns:
        work_df["GRID_Z_SCORE"] = 0.0
    if "ANOMALY_SCORE" not in work_df.columns:
        work_df["ANOMALY_SCORE"] = 0.2
    if "IRRADIATION" not in work_df.columns:
        work_df["IRRADIATION"] = 0.0

    feature_cols = [
        power_col, "MODEL_RESIDUAL", "GRID_Z_SCORE",
        "PEER_RANK_PCT", "ANOMALY_SCORE", "IRRADIATION"
    ]

    snapshots = []
    inv_to_idx = {inv: idx for idx, inv in enumerate(inverter_list)}
    num_nodes = len(inverter_list)

    for dt, grp in work_df.groupby(timestamp_col):
        feat_matrix = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)
        y_vec = np.zeros((num_nodes, 1), dtype=np.float32)

        for _, row in grp.iterrows():
            inv = row[inverter_col]
            if inv in inv_to_idx:
                idx = inv_to_idx[inv]
                p_kw = float(row[power_col])
                # Clip extreme z-scores to [-3, 3] for graph stability
                raw_z = float(row["GRID_Z_SCORE"])
                clip_z = max(-3.0, min(3.0, raw_z))
                
                feats = [
                    p_kw / 1000.0,
                    float(row["MODEL_RESIDUAL"]) / 100.0,
                    clip_z,
                    float(row["PEER_RANK_PCT"]),
                    float(row["ANOMALY_SCORE"]),
                    float(row["IRRADIATION"]),
                ]
                feat_matrix[idx] = feats
                y_vec[idx, 0] = p_kw / 1000.0

        snapshots.append({
            "DATE_TIME": dt,
            "x": torch.tensor(feat_matrix, dtype=torch.float32),
            "y": torch.tensor(y_vec, dtype=torch.float32),
        })

    return snapshots


def train_gat_model(
    model: InverterGAT,
    snapshots: List[Dict[str, Any]],
    edge_index: torch.Tensor,
    epochs: int = 10,
    lr: float = 0.001,
) -> List[float]:
    """
    Self-supervised training loop: predicts node AC power from fleet graph neighbors.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    epoch_losses = []

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for snap in snapshots:
            optimizer.zero_grad()
            out, _ = model(snap["x"], edge_index)
            loss = loss_fn(out, snap["y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(snapshots))
        epoch_losses.append(avg_loss)

    return epoch_losses


def predict_gat_anomalies(
    model: InverterGAT,
    snapshots: List[Dict[str, Any]],
    edge_index: torch.Tensor,
    inverter_list: List[str],
) -> pd.DataFrame:
    """
    Computes spatial reconstruction error per inverter node across snapshots.
    """
    model.eval()
    records = []

    with torch.no_grad():
        for snap in snapshots:
            dt = snap["DATE_TIME"]
            out, _ = model(snap["x"], edge_index)
            actual = snap["y"].numpy() * 1000.0  # re-scale back to kW
            pred = out.numpy() * 1000.0
            error = np.abs(actual - pred).flatten()

            for i, inv in enumerate(inverter_list):
                records.append({
                    "DATE_TIME": dt,
                    "INVERTER_ID": inv,
                    "GAT_ACTUAL_POWER": float(actual[i, 0]),
                    "GAT_PREDICTED_POWER": float(pred[i, 0]),
                    "GAT_RECONSTRUCTION_ERROR": float(error[i]),
                })

    return pd.DataFrame(records)
