"""Traffic-state-aware mixture decoder for the pure DSTSGCN backbone."""
from __future__ import annotations

import torch
from torch import nn


class StatisticalMixtureForecastWrapper(nn.Module):
    """Fuse direct, autoregressive and persistence experts with a robust state gate."""

    expert_names = ("direct", "autoregressive", "persistence")

    def __init__(
        self,
        base_model: nn.Module,
        hidden_dim: int,
        horizon: int,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.graph_learner = base_model.graph_learner
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)
        self.input_projection = nn.Linear(self.output_dim, self.hidden_dim)
        self.decoder_cell = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)
        gate_input_dim = self.hidden_dim + 3 * self.output_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.horizon * len(self.expert_names)),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def _autoregressive_forecast(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, hidden_dim = hidden.shape
        decoder_input = x[:, -1, :, : self.output_dim]
        outputs = []
        for _ in range(self.horizon):
            embedded = torch.tanh(self.input_projection(decoder_input))
            hidden = self.decoder_cell(
                embedded.reshape(batch_size * num_nodes, hidden_dim),
                hidden.reshape(batch_size * num_nodes, hidden_dim),
            ).reshape(batch_size, num_nodes, hidden_dim)
            decoder_input = self.output_projection(hidden)
            outputs.append(decoder_input)
        return torch.stack(outputs, dim=1), hidden

    def _traffic_state_features(self, x: torch.Tensor) -> torch.Tensor:
        history = x[..., : self.output_dim]
        history_median = history.median(dim=1).values
        absolute_deviation = torch.abs(history - history_median.unsqueeze(1))
        history_mad = absolute_deviation.median(dim=1).values.clamp_min(1e-3)
        robust_level = (history[:, -1] - history_median) / history_mad
        trend = (history[:, -1] - history[:, 0]) / max(history.shape[1] - 1, 1)
        if history.shape[1] > 1:
            volatility = torch.abs(history[:, 1:] - history[:, :-1]).median(dim=1).values
        else:
            volatility = torch.zeros_like(history[:, -1])
        return torch.cat((robust_level, trend, volatility), dim=-1)

    def forward(self, x: torch.Tensor, static_adj: torch.Tensor, *args, **kwargs):
        requested_aux = bool(kwargs.get("return_aux", False))
        kwargs["return_aux"] = True
        direct_prediction, aux = self.base_model(x, static_adj, *args, **kwargs)
        hidden = aux["final_state"]
        if hidden.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, received {hidden.shape[-1]}"
            )
        autoregressive_prediction, decoder_final_state = self._autoregressive_forecast(x, hidden)
        persistence_prediction = x[:, -1:, :, : self.output_dim].expand(
            -1, self.horizon, -1, -1
        )
        traffic_state = self._traffic_state_features(x)
        gate_input = torch.cat((hidden, traffic_state), dim=-1)
        batch_size, num_nodes, _ = gate_input.shape
        gate_logits = self.gate(gate_input).reshape(
            batch_size, num_nodes, self.horizon, len(self.expert_names)
        )
        mixture_weights = torch.softmax(gate_logits.permute(0, 2, 1, 3), dim=-1)
        expert_predictions = torch.stack(
            (direct_prediction, autoregressive_prediction, persistence_prediction), dim=-1
        )
        prediction = (
            expert_predictions
            * mixture_weights.unsqueeze(-2)
        ).sum(dim=-1)
        if requested_aux:
            aux["decoder_final_state"] = decoder_final_state
            aux["expert_predictions"] = expert_predictions
            aux["mixture_weights"] = mixture_weights
            aux["traffic_state_features"] = traffic_state
            return prediction, aux
        return prediction
