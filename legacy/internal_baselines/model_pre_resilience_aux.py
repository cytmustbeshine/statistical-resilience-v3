"""Mainline DGCN-STSGCN models for the thesis experiments.

This file intentionally keeps only the thesis mainline:
  1. STSGCN-style static graph baseline through fusion_mode="static"
  2. DGCN-style dynamic graph baseline through fusion_mode="dynamic"
  3. quality_v2 through fusion_type="quality"
  4. quality_ood_scaled_v1 through fusion_type="quality_ood_scaled"
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def row_normalize(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-normalize an adjacency matrix."""
    return adj / (adj.sum(dim=-1, keepdim=True) + eps)


def add_self_loop(adj: torch.Tensor) -> torch.Tensor:
    """Add self loops to [N, N] or [B, N, N] adjacency matrices."""
    n = adj.size(-1)
    eye = torch.eye(n, device=adj.device, dtype=adj.dtype)
    if adj.dim() == 2:
        return adj + eye
    return adj + eye.unsqueeze(0)


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Add self loops and row-normalize an adjacency matrix once."""
    return row_normalize(add_self_loop(adj))


def scalar_to_logit(value: float) -> torch.Tensor:
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return torch.logit(torch.tensor(value, dtype=torch.float32))


def topk_sparsify(adj: torch.Tensor, k: int | None) -> torch.Tensor:
    """Keep only top-k outgoing weights for each node."""
    if k is None or k <= 0 or k >= adj.size(-1):
        return adj
    values, indices = torch.topk(adj, k=k, dim=-1)
    sparse = torch.zeros_like(adj)
    sparse.scatter_(-1, indices, values)
    return sparse


class DynamicGraphLearner(nn.Module):
    """Attention-based fallback dynamic graph learner."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        top_k: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.top_k = top_k
        self.temporal_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, history_steps, num_nodes, input_dim = x.shape
        node_series = x.permute(0, 2, 1, 3).reshape(
            batch_size * num_nodes,
            history_steps,
            input_dim,
        )
        _, hidden = self.temporal_encoder(node_series)
        node_embed = hidden[-1].reshape(batch_size, num_nodes, -1)
        q = self.query(node_embed).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.key(node_embed).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        dynamic_adj = topk_sparsify(attn.mean(dim=1), self.top_k)
        return normalize_adjacency(dynamic_adj)


class LaplaceMatrixLatentNetwork(nn.Module):
    """DGCN-style latent dynamic graph learner."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        top_k: int | None = None,
        matrix_hidden_dim: int | None = 256,
        static_adj_is_normalized: bool = False,
        num_diffusion_steps: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        matrix_size = num_nodes * num_nodes
        if matrix_hidden_dim is None or matrix_hidden_dim <= 0:
            matrix_hidden_dim = matrix_size
        self.matrix_hidden_dim = int(matrix_hidden_dim)
        self.static_adj_is_normalized = static_adj_is_normalized
        self.num_diffusion_steps = max(1, int(num_diffusion_steps))
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.matrix_lstm = nn.LSTM(matrix_size, self.matrix_hidden_dim, batch_first=True)
        self.lstm_proj = (
            None
            if self.matrix_hidden_dim == matrix_size
            else nn.Linear(self.matrix_hidden_dim, matrix_size)
        )
        self.global_residual = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)

    def _spatial_attention(self, x_t: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x_t.shape
        h = torch.tanh(self.feature_proj(x_t))
        q = self.query(h).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.key(h).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        return self.dropout(torch.softmax(scores, dim=-1).mean(dim=1))

    def forward(
        self,
        x: torch.Tensor,
        static_adj: torch.Tensor | None = None,
        static_adj_is_normalized: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, history_steps, num_nodes, _ = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, received {num_nodes} nodes.")

        dynamic_sequence = torch.stack(
            [self._spatial_attention(x[:, t]) for t in range(history_steps)],
            dim=1,
        )
        temporal_reg = x.new_tensor(0.0)
        if self.training and history_steps > 1:
            temporal_reg = ((dynamic_sequence[:, 1:] - dynamic_sequence[:, :-1]) ** 2).mean()

        matrix_series = dynamic_sequence.reshape(batch_size, history_steps, -1)
        lstm_out, _ = self.matrix_lstm(matrix_series)
        last_state = lstm_out[:, -1]
        if self.lstm_proj is not None:
            last_state = self.lstm_proj(last_state)
        dynamic_matrix = torch.sigmoid(last_state.reshape(batch_size, num_nodes, num_nodes))

        if static_adj is None:
            base = torch.zeros(num_nodes, num_nodes, device=x.device, dtype=x.dtype)
        else:
            base = static_adj.to(device=x.device, dtype=x.dtype)
            is_normalized = (
                self.static_adj_is_normalized
                if static_adj_is_normalized is None
                else static_adj_is_normalized
            )
            if not is_normalized:
                base = normalize_adjacency(base)

        residual = torch.sigmoid(self.global_residual)
        global_graph = normalize_adjacency(base + residual)
        dynamic_adj = dynamic_matrix * global_graph.unsqueeze(0)
        dynamic_adj = topk_sparsify(dynamic_adj, self.top_k)
        dynamic_adj = normalize_adjacency(dynamic_adj)

        if self.num_diffusion_steps > 1:
            diffused = dynamic_adj.clone()
            power = dynamic_adj.clone()
            alpha = 0.5
            for _ in range(self.num_diffusion_steps - 1):
                power = torch.bmm(power, dynamic_adj)
                diffused = diffused + alpha * power
                alpha *= 0.5
            dynamic_adj = row_normalize(diffused)
        return dynamic_adj, temporal_reg


class GraphFusion(nn.Module):
    """quality_v2 fusion with optional OOD residual scaling."""

    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        init_static_weight: float = 0.5,
        mode: str = "fusion",
        fusion_type: str = "quality",
        quality_gate_bias: float = -1.0,
        static_adj_is_normalized: bool = False,
        event_dim: int = 0,
        residual_init: float = 0.2,
        ood_gamma_max: float = 0.35,
    ) -> None:
        super().__init__()
        if mode not in {"fusion", "static", "dynamic"}:
            raise ValueError("fusion mode must be one of: 'fusion', 'static', 'dynamic'.")
        if fusion_type not in {"quality", "quality_ood_scaled"}:
            raise ValueError("fusion_type must be 'quality' or 'quality_ood_scaled'.")
        self.mode = mode
        self.fusion_type = fusion_type
        self.event_dim = max(0, int(event_dim))
        self.static_adj_is_normalized = static_adj_is_normalized
        self.ood_gamma_max = max(float(ood_gamma_max), 0.0)
        if mode == "fusion":
            self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(init_static_weight)))
            self.residual_logit = nn.Parameter(scalar_to_logit(residual_init))
            self.beta = nn.Parameter(torch.tensor(0.3))
            gate_hidden = max(8, hidden_dim // 2)
            self.quality_gate: nn.Module | None = nn.Sequential(
                nn.Linear(hidden_dim + 1 + self.event_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.constant_(self.quality_gate[-1].bias, quality_gate_bias)
        else:
            self.register_buffer("alpha_logit", torch.logit(torch.tensor(init_static_weight)))
            self.register_buffer("residual_logit", scalar_to_logit(residual_init))
            self.register_buffer("beta", torch.tensor(0.3))
            self.quality_gate = None
        self.last_residual_reg: torch.Tensor | None = None
        self.last_ood_intensity_mean: torch.Tensor | None = None
        self.last_ood_intensity_std: torch.Tensor | None = None
        self.last_ood_scale_mean: torch.Tensor | None = None
        self.last_ood_scale_std: torch.Tensor | None = None

    def forward(
        self,
        static_adj: torch.Tensor,
        dynamic_adj: torch.Tensor,
        node_context: torch.Tensor | None = None,
        static_adj_is_normalized: bool | None = None,
        event_signal: torch.Tensor | None = None,
        ood_intensity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_normalized = (
            self.static_adj_is_normalized
            if static_adj_is_normalized is None
            else static_adj_is_normalized
        )
        static_norm = static_adj if is_normalized else normalize_adjacency(static_adj)
        dynamic_norm = normalize_adjacency(dynamic_adj)
        self.last_residual_reg = dynamic_norm.new_tensor(0.0)
        self.last_ood_intensity_mean = dynamic_norm.new_tensor(0.0)
        self.last_ood_intensity_std = dynamic_norm.new_tensor(0.0)
        self.last_ood_scale_mean = dynamic_norm.new_tensor(1.0)
        self.last_ood_scale_std = dynamic_norm.new_tensor(0.0)

        if self.mode == "static":
            return static_norm.unsqueeze(0).expand_as(dynamic_norm)
        if self.mode == "dynamic":
            return dynamic_norm

        static_prior = static_norm.unsqueeze(0).expand_as(dynamic_norm)
        s_flat = static_prior.reshape(dynamic_norm.size(0), dynamic_norm.size(1), -1)
        d_flat = dynamic_norm.reshape(dynamic_norm.size(0), dynamic_norm.size(1), -1)
        cos_sim = F.cosine_similarity(s_flat, d_flat, dim=-1, eps=1e-6)
        disagreement = (1.0 - cos_sim).clamp(min=0.0, max=2.0).unsqueeze(-1)

        if node_context is None:
            weight = torch.sigmoid(self.beta) * disagreement
        else:
            gate_parts = [node_context, disagreement]
            if self.event_dim > 0:
                if event_signal is None:
                    ev = torch.zeros(
                        node_context.size(0),
                        self.event_dim,
                        device=node_context.device,
                        dtype=node_context.dtype,
                    )
                else:
                    ev = event_signal.to(device=node_context.device, dtype=node_context.dtype)
                    if ev.dim() == 1:
                        ev = ev.unsqueeze(-1)
                ev = ev[:, : self.event_dim]
                if ev.size(-1) < self.event_dim:
                    pad = torch.zeros(
                        ev.size(0),
                        self.event_dim - ev.size(-1),
                        device=ev.device,
                        dtype=ev.dtype,
                    )
                    ev = torch.cat([ev, pad], dim=-1)
                gate_parts.append(ev.unsqueeze(1).expand(-1, node_context.size(1), -1))
            if self.quality_gate is None:
                raise RuntimeError("quality gate is only available in fusion mode.")
            weight = torch.sigmoid(self.quality_gate(torch.cat(gate_parts, dim=-1)))

        ood_scale: torch.Tensor | float = 1.0
        if self.fusion_type == "quality_ood_scaled":
            if ood_intensity is None:
                ood_score = torch.zeros(
                    dynamic_norm.size(0),
                    1,
                    1,
                    device=dynamic_norm.device,
                    dtype=dynamic_norm.dtype,
                )
            else:
                ood_score = ood_intensity.to(device=dynamic_norm.device, dtype=dynamic_norm.dtype)
                if ood_score.dim() == 1:
                    ood_score = ood_score.view(-1, 1, 1)
                elif ood_score.dim() == 2:
                    ood_score = ood_score.unsqueeze(-1)
                elif ood_score.dim() != 3:
                    raise ValueError("ood_intensity must have shape [B], [B, 1], or [B, 1, 1].")
                if ood_score.size(0) != dynamic_norm.size(0):
                    raise ValueError("ood_intensity batch size does not match dynamic adjacency.")
                ood_score = ood_score[:, :1, :1]
            ood_score = ood_score.clamp(0.0, 1.0)
            ood_scale = 1.0 + self.ood_gamma_max * ood_score
            self.last_ood_intensity_mean = ood_score.mean()
            self.last_ood_intensity_std = ood_score.std(unbiased=False)
            self.last_ood_scale_mean = ood_scale.mean()
            self.last_ood_scale_std = ood_scale.std(unbiased=False)

        fused = static_prior + ood_scale * weight * (dynamic_norm - static_prior)
        return row_normalize(torch.clamp(fused, min=0.0))

    def residual_gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.residual_logit)


class SpatialTemporalSynchronousConv(nn.Module):
    """STSGCN-style localized synchronous graph convolution."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        activation: str = "glu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_layers):
            out_dim = hidden_dim * 2 if activation == "glu" else hidden_dim
            self.layers.append(nn.Linear(in_dim, out_dim))
            in_dim = hidden_dim

    def _build_local_adj(self, fused_adj: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = fused_adj.shape
        local_adj = torch.zeros(
            batch_size,
            3 * num_nodes,
            3 * num_nodes,
            device=fused_adj.device,
            dtype=fused_adj.dtype,
        )
        for i in range(3):
            s = i * num_nodes
            e = (i + 1) * num_nodes
            local_adj[:, s:e, s:e] = fused_adj
        eye = torch.eye(num_nodes, device=fused_adj.device, dtype=fused_adj.dtype)
        for i in range(2):
            a = i * num_nodes
            b = (i + 1) * num_nodes
            local_adj[:, a : a + num_nodes, b : b + num_nodes] = eye
            local_adj[:, b : b + num_nodes, a : a + num_nodes] = eye
        return row_normalize(add_self_loop(local_adj))

    def forward(self, x_window: torch.Tensor, fused_adj: torch.Tensor) -> torch.Tensor:
        batch_size, window_size, num_nodes, input_dim = x_window.shape
        if window_size != 3:
            raise ValueError("SpatialTemporalSynchronousConv expects 3 time steps.")
        h = x_window.reshape(batch_size, 3 * num_nodes, input_dim)
        local_adj = self._build_local_adj(fused_adj)
        layer_outputs = []
        for layer in self.layers:
            h = torch.bmm(local_adj, h)
            h = layer(h)
            if self.activation == "glu":
                h_left, h_gate = h.chunk(2, dim=-1)
                h = h_left * torch.sigmoid(h_gate)
            else:
                h = F.relu(h)
            h = self.dropout(h)
            layer_outputs.append(h)
        h = torch.stack(layer_outputs, dim=0).max(dim=0).values
        return h[:, num_nodes : 2 * num_nodes, :]


class DSTSGCNBlock(nn.Module):
    """One dynamic STSGCN block over all sliding local windows."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_gcn_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stsgcm = SpatialTemporalSynchronousConv(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gcn_layers,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, fused_adj: torch.Tensor) -> torch.Tensor:
        outputs = []
        for start in range(x.size(1) - 2):
            h = self.stsgcm(x[:, start : start + 3], fused_adj)
            outputs.append(self.norm(h))
        return torch.stack(outputs, dim=1)


class DSTSGCN(nn.Module):
    """Dynamic graph learning plus STSGCN forecasting."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int = 1,
        horizon: int = 12,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        graph_learner_type: str = "lmln",
        fusion_mode: str = "fusion",
        fusion_type: str = "quality",
        dynamic_top_k: int | None = None,
        quality_gate_bias: float = -1.0,
        matrix_hidden_dim: int | None = 256,
        static_adj_is_normalized: bool = False,
        use_learned_static: bool = False,
        num_diffusion_steps: int = 2,
        event_dim: int = 0,
        graph_residual_init: float = 0.2,
        ood_gamma_max: float = 0.35,
    ) -> None:
        super().__init__()
        if use_learned_static:
            raise ValueError("The clean thesis model does not support learned static fusion.")
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.output_dim = output_dim
        self.static_adj_is_normalized = static_adj_is_normalized
        self.fusion_mode = fusion_mode
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        if fusion_mode == "static":
            self.graph_learner_type = "none"
            self.graph_learner = None
        elif graph_learner_type == "attention":
            self.graph_learner_type = graph_learner_type
            self.graph_learner = DynamicGraphLearner(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                top_k=dynamic_top_k,
            )
        elif graph_learner_type == "lmln":
            self.graph_learner_type = graph_learner_type
            self.graph_learner = LaplaceMatrixLatentNetwork(
                num_nodes=num_nodes,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                top_k=dynamic_top_k,
                matrix_hidden_dim=matrix_hidden_dim,
                static_adj_is_normalized=static_adj_is_normalized,
                num_diffusion_steps=num_diffusion_steps,
            )
        else:
            raise ValueError("graph_learner_type must be one of: 'lmln', 'attention'.")
        self.graph_fusion = GraphFusion(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            mode=fusion_mode,
            fusion_type=fusion_type,
            quality_gate_bias=quality_gate_bias,
            static_adj_is_normalized=static_adj_is_normalized,
            event_dim=event_dim,
            residual_init=graph_residual_init,
            ood_gamma_max=ood_gamma_max,
        )
        self.blocks = nn.ModuleList(
            [
                DSTSGCNBlock(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        static_adj: torch.Tensor,
        road_adj: torch.Tensor | None = None,
        event_signal: torch.Tensor | None = None,
        ood_intensity: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        current_static_is_normalized = self.static_adj_is_normalized
        if self.fusion_mode == "static":
            dynamic_adj = static_adj.unsqueeze(0).expand(x.size(0), -1, -1)
            temporal_reg = x.new_tensor(0.0)
        elif self.graph_learner_type == "lmln":
            graph_base = None if self.fusion_mode == "dynamic" else static_adj
            if self.graph_learner is None:
                raise RuntimeError("DGCN graph learner is not initialized.")
            dynamic_adj, temporal_reg = self.graph_learner(
                x,
                graph_base,
                static_adj_is_normalized=current_static_is_normalized,
            )
        else:
            if self.graph_learner is None:
                raise RuntimeError("Dynamic graph learner is not initialized.")
            dynamic_adj = self.graph_learner(x)
            temporal_reg = x.new_tensor(0.0)

        h = self.input_proj(x)
        fused_adj = self.graph_fusion(
            static_adj,
            dynamic_adj,
            node_context=h[:, -1],
            static_adj_is_normalized=current_static_is_normalized,
            event_signal=event_signal,
            ood_intensity=ood_intensity,
        )
        graph_residual_reg = self.graph_fusion.last_residual_reg or x.new_tensor(0.0)
        ood_intensity_mean = self.graph_fusion.last_ood_intensity_mean or x.new_tensor(0.0)
        ood_intensity_std = self.graph_fusion.last_ood_intensity_std or x.new_tensor(0.0)
        ood_scale_mean = self.graph_fusion.last_ood_scale_mean or x.new_tensor(1.0)
        ood_scale_std = self.graph_fusion.last_ood_scale_std or x.new_tensor(0.0)

        for block in self.blocks:
            if h.size(1) < 3:
                raise ValueError("History length is too short for stacked STSGCN blocks.")
            h = self.dropout(block(h, fused_adj))

        pred = self.output(h[:, -1])
        pred = pred.view(x.size(0), self.num_nodes, self.horizon, self.output_dim)
        pred = pred.permute(0, 2, 1, 3).contiguous()
        if not return_aux:
            return pred
        return pred, {
            "temporal_reg": temporal_reg,
            "graph_residual_reg": graph_residual_reg,
            "graph_residual_gamma": self.graph_fusion.residual_gamma(),
            "ood_intensity_mean": ood_intensity_mean,
            "ood_intensity_std": ood_intensity_std,
            "ood_scale_mean": ood_scale_mean,
            "ood_scale_std": ood_scale_std,
        }
