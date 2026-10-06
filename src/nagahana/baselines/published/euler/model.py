"""EULER's network: a per-snapshot GNN encoder, a recurrent model over snapshots, an inner-product decoder.

For snapshot t with propagation operator A_t (graph.py):

    GCN encoder     H_1 = ReLU(A_t E)                    E [V, hidden]: one-hot node features times the
                                                         first layer's weights, i.e. a node table
                    H_l = ReLU(A_t H_{l-1} W_l)          l = 2 ... L - 1
                    X_t = tanh(A_t H_{L-1} W_L)          [V, d_gnn]
    SAGE encoder    the same stack with mean aggregation (PyTorch Geometric's SAGEConv):
                    H_l = act(W_self H_i + W_neigh mean_{j -> i} H_j + b); on the one-hot input of the
                    first layer W_self and W_neigh are two node tables
    recurrent       h_t = GRU(X_t, h_{t-1}) (or LSTM), per node; Z_t = W_out h_t   [V, d_z]
    decoder         P(edge u -> v) = sigmoid(<Z_u, Z_v>)

Link prediction scores the edges of snapshot t + 1 with Z_t; link detection scores the edges of
snapshot t with Z_t (the encoder of snapshot t sees them). Dropout acts on the encoder inputs of every
layer during training. Layer sizes and activations: AS-552 (citation to verify).
"""

from __future__ import annotations

import torch
from torch import nn

from nagahana.baselines.published.euler.graph import mean_neighbours, propagate


class SnapshotEncoder(nn.Module):
    """GCN or GraphSAGE stack over one snapshot (module docstring)."""

    def __init__(self, kind: str, n_nodes: int, hidden: int, out_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        if kind not in ("gcn", "sage"):
            raise ValueError("encoder must be 'gcn' or 'sage'")
        if layers < 1:
            raise ValueError("the encoder needs at least one layer")
        self.kind = kind
        self.node_table = nn.Parameter(torch.empty(n_nodes, hidden if layers > 1 else out_dim))
        nn.init.xavier_uniform_(self.node_table)
        with torch.no_grad():
            self.node_table[0].zero_()                                  # row 0: nodes unseen in training
        dims = [hidden] * (layers - 1) + [out_dim]
        self.layers = nn.ModuleList()
        self.self_layers = nn.ModuleList()
        for i in range(1, layers):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))
            if kind == "sage":
                self.self_layers.append(nn.Linear(dims[i - 1], dims[i], bias=False))
        # GraphSAGE on one-hot inputs: W_self x_i and W_neigh mean_j x_j are two independent node tables.
        self.neigh_table = nn.Parameter(torch.empty_like(self.node_table)) if kind == "sage" else None
        self.neigh_bias = nn.Parameter(torch.zeros(self.node_table.shape[1])) if kind == "sage" else None
        if self.neigh_table is not None:
            nn.init.xavier_uniform_(self.neigh_table)
            with torch.no_grad():
                self.neigh_table[0].zero_()
        self.drop = nn.Dropout(dropout)
        self.n_layers = layers

    def forward(self, rows: torch.Tensor, cols: torch.Tensor, values: torch.Tensor, self_w: torch.Tensor,
                src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Node features [V, out_dim] of one snapshot (rows/cols/values/self_w: the GCN operator)."""
        x = self.node_table
        for li in range(self.n_layers):
            last = li == self.n_layers - 1
            if self.kind == "gcn":
                h = propagate(self.drop(x), rows, cols, values, self_w)
                if li > 0:
                    h = self.layers[li - 1](h)
            elif li == 0:
                assert self.neigh_table is not None and self.neigh_bias is not None
                h = self.drop(x) + mean_neighbours(self.drop(self.neigh_table), src, dst) + self.neigh_bias
            else:
                xin = self.drop(x)
                h = self.self_layers[li - 1](xin) + self.layers[li - 1](mean_neighbours(xin, src, dst))
            x = torch.tanh(h) if last else torch.relu(h)
        return x


class EulerNet(nn.Module):
    """Snapshot encoder + recurrent model + linear projection to the link embedding."""

    def __init__(self, *, encoder: str, rnn: str, n_nodes: int, hidden: int, gnn_out: int, rnn_hidden: int, embed: int,
                 layers: int, dropout: float) -> None:
        super().__init__()
        self.encoder = SnapshotEncoder(encoder, n_nodes, hidden, gnn_out, layers, dropout)
        if rnn == "gru":
            self.rnn: nn.GRUCell | nn.LSTMCell = nn.GRUCell(gnn_out, rnn_hidden)
        elif rnn == "lstm":
            self.rnn = nn.LSTMCell(gnn_out, rnn_hidden)
        else:
            raise ValueError("rnn must be 'gru' or 'lstm'")
        self.out = nn.Linear(rnn_hidden, embed)
        self.rnn_kind = rnn
        self.rnn_hidden = rnn_hidden

    def initial_state(self, n_nodes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero hidden (and cell) state [V, rnn_hidden]."""
        z = torch.zeros(n_nodes, self.rnn_hidden, device=device)
        return z, z.clone()

    def step(self, graph: tuple[torch.Tensor, ...], state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One snapshot: (Z_t [V, embed], new state)."""
        x = self.encoder(*graph)
        h, c = state
        if self.rnn_kind == "gru":
            assert isinstance(self.rnn, nn.GRUCell)
            h = self.rnn(x, h)
        else:
            assert isinstance(self.rnn, nn.LSTMCell)
            h, c = self.rnn(x, (h, c))
        return self.out(h), (h, c)


def edge_logits(z: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """<Z_u, Z_v> for every edge (logit of the edge probability)."""
    return (z[src] * z[dst]).sum(dim=-1)
