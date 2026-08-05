"""Pure DSTSGCN autoregressive temporal decoder."""
from __future__ import annotations

import torch
from torch import nn


class AutoregressiveForecastWrapper(nn.Module):
    """Decode DSTSGCN node states one horizon step at a time."""

    def __init__(self, base_model: nn.Module, hidden_dim: int, horizon: int, output_dim: int = 1):
        super().__init__()
        self.base_model = base_model
        self.graph_learner = base_model.graph_learner
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.output_dim = int(output_dim)
        self.input_projection = nn.Linear(self.output_dim, self.hidden_dim)
        self.decoder_cell = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        static_adj: torch.Tensor,
        *args,
        labels: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        **kwargs,
    ):
        if not 0.0 <= float(teacher_forcing_ratio) <= 1.0:
            raise ValueError("teacher_forcing_ratio must be between 0 and 1")
        if labels is not None:
            expected = (x.shape[0], self.horizon, x.shape[2], self.output_dim)
            if tuple(labels.shape) != expected:
                raise ValueError(f"Expected labels shape {expected}, received {tuple(labels.shape)}")
        requested_aux = bool(kwargs.get("return_aux", False))
        kwargs["return_aux"] = True
        _, aux = self.base_model(x, static_adj, *args, **kwargs)
        hidden = aux["final_state"]
        batch_size, num_nodes, hidden_dim = hidden.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, received {hidden_dim}")
        decoder_input = x[:, -1, :, : self.output_dim]
        outputs = []
        for step in range(self.horizon):
            embedded = torch.tanh(self.input_projection(decoder_input))
            hidden = self.decoder_cell(
                embedded.reshape(batch_size * num_nodes, hidden_dim),
                hidden.reshape(batch_size * num_nodes, hidden_dim),
            ).reshape(batch_size, num_nodes, hidden_dim)
            prediction_step = self.output_projection(hidden)
            outputs.append(prediction_step)
            decoder_input = prediction_step
            if self.training and labels is not None and step + 1 < self.horizon:
                target_step = labels[:, step, :, : self.output_dim]
                finite_target = torch.isfinite(target_step)
                if teacher_forcing_ratio >= 1.0:
                    use_target = finite_target
                elif teacher_forcing_ratio > 0.0:
                    sampled = torch.rand_like(target_step) < float(teacher_forcing_ratio)
                    use_target = sampled & finite_target
                else:
                    use_target = torch.zeros_like(finite_target)
                decoder_input = torch.where(use_target, target_step, prediction_step)
        prediction = torch.stack(outputs, dim=1)
        if requested_aux:
            aux["decoder_final_state"] = hidden
            return prediction, aux
        return prediction
