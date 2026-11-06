import pytest
import numpy as np
import torch
import pandas as pd
from src.models.graph_network import (
    build_adjacency_from_correlation,
    InverterGAT,
    prepare_fleet_snapshots,
    train_gat_model,
    predict_gat_anomalies,
)

class TestGraphNetwork:
    def test_correlation_adjacency(self, synthetic_fleet_df):
        """Correlation adjacency must be symmetric, zero-diagonal, with valid edge_index."""
        adj_matrix, edge_index, inv_list = build_adjacency_from_correlation(
            synthetic_fleet_df, threshold=0.70, value_col="AC_POWER"
        )
        assert isinstance(adj_matrix, np.ndarray)
        assert adj_matrix.shape == (22, 22)
        # Symmetry
        assert np.allclose(adj_matrix, adj_matrix.T)
        # No self loops in adj
        assert np.all(np.diag(adj_matrix) == 0)
        # edge_index is LongTensor of shape (2, E)
        assert isinstance(edge_index, torch.Tensor)
        assert edge_index.ndim == 2
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0, "Graph must have connected edges"
        assert len(inv_list) == 22

    def test_inverter_gat_forward_shape(self):
        """InverterGAT forward pass must map (N, F_in) nodes to (N, 1) power prediction."""
        num_nodes = 22
        in_features = 6
        x = torch.randn(num_nodes, in_features)
        
        # Ring topology edges for testing
        src = torch.arange(num_nodes)
        dst = (src + 1) % num_nodes
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        
        model = InverterGAT(in_features=in_features, hidden_dim=32, num_heads=4)
        out, att_weights = model(x, edge_index, return_attention=True)
        
        assert out.shape == (num_nodes, 1)
        assert not torch.isnan(out).any()
        assert att_weights is not None
        assert att_weights.shape[0] == edge_index.shape[1]

    def test_gat_training_loss_decrease(self, synthetic_fleet_df):
        """Training GAT model for a few epochs must reduce reconstruction MSE loss."""
        adj_matrix, edge_index, inv_list = build_adjacency_from_correlation(
            synthetic_fleet_df, threshold=0.70
        )
        snapshots = prepare_fleet_snapshots(synthetic_fleet_df, inv_list)
        assert len(snapshots) == 96
        
        model = InverterGAT(in_features=6, hidden_dim=16, num_heads=2)
        losses = train_gat_model(model, snapshots, edge_index, epochs=5, lr=0.01)
        
        assert len(losses) == 5
        assert losses[-1] < losses[0], f"Loss should decrease: initial {losses[0]:.4f}, final {losses[-1]:.4f}"

    def test_bad_inverter_spatial_error(self, synthetic_fleet_df):
        """INV_BAD must produce significantly higher spatial reconstruction error than healthy peers."""
        adj_matrix, edge_index, inv_list = build_adjacency_from_correlation(
            synthetic_fleet_df, threshold=0.70
        )
        snapshots = prepare_fleet_snapshots(synthetic_fleet_df, inv_list)
        
        model = InverterGAT(in_features=6, hidden_dim=16, num_heads=2)
        train_gat_model(model, snapshots, edge_index, epochs=6, lr=0.01)
        
        errors_df = predict_gat_anomalies(model, snapshots, edge_index, inv_list)
        assert isinstance(errors_df, pd.DataFrame)
        assert "INVERTER_ID" in errors_df.columns
        assert "GAT_RECONSTRUCTION_ERROR" in errors_df.columns
        
        # Aggregate error per inverter
        mean_errors = errors_df.groupby("INVERTER_ID")["GAT_RECONSTRUCTION_ERROR"].mean()
        bad_err = mean_errors["INV_BAD"]
        healthy_avg = mean_errors[mean_errors.index != "INV_BAD"].mean()
        
        assert bad_err > healthy_avg * 1.5, (
            f"INV_BAD spatial error ({bad_err:.4f}) should be > 1.5x healthy average ({healthy_avg:.4f})"
        )
