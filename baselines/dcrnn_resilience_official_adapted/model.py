"""High-fidelity PyTorch adaptation of the published DCRNN-Resilience code.

The implementation follows the public TensorFlow source from
Charles117/resilience_shenzhen: dual random-walk supports, Chebyshev diffusion,
DCGRU encoder-decoder, inverse-sigmoid curriculum learning, and N-to-M input
and output handling. Private Shenzhen data and unpublished resilience code are
not reproduced here.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def random_walk_supports(adj: torch.Tensor) -> list[torch.Tensor]:
    """Build forward and reverse row-normalized random-walk matrices."""
    adj = torch.clamp(adj, min=0.0)
    forward = adj / torch.clamp(adj.sum(-1, keepdim=True), min=1e-6)
    reverse_adj = adj.transpose(-1, -2)
    reverse = reverse_adj / torch.clamp(reverse_adj.sum(-1, keepdim=True), min=1e-6)
    return [forward, reverse]


class DiffusionGraphConv(nn.Module):
    """Official-style Chebyshev diffusion over one or more graph supports."""

    def __init__(self, input_dim: int, output_dim: int, max_diffusion_step: int = 1, bias_start: float = 0.0):
        super().__init__()
        self.max_diffusion_step = max(0, int(max_diffusion_step))
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        # dual_random_walk has two supports; x0 is included exactly once.
        matrices = 1 + 2 * self.max_diffusion_step
        self.weight = nn.Parameter(torch.empty(self.input_dim * matrices, self.output_dim))
        self.bias = nn.Parameter(torch.full((self.output_dim,), float(bias_start)))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        features = [x]
        for support in supports:
            x0 = x
            if self.max_diffusion_step >= 1:
                x1 = torch.einsum("nm,bmf->bnf", support, x0)
                features.append(x1)
            for _ in range(2, self.max_diffusion_step + 1):
                x2 = 2.0 * torch.einsum("nm,bmf->bnf", support, x1) - x0
                features.append(x2)
                x1, x0 = x2, x1
        stacked = torch.cat(features, dim=-1)
        return torch.einsum("bnf,fo->bno", stacked, self.weight) + self.bias


class OfficialDCGRUCell(nn.Module):
    """DCGRU cell matching the public resilience_shenzhen gate equations."""

    def __init__(self, input_dim: int, hidden_dim: int, max_diffusion_step: int = 1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        joined = int(input_dim) + self.hidden_dim
        self.gates = DiffusionGraphConv(joined, 2 * self.hidden_dim, max_diffusion_step, bias_start=1.0)
        self.candidate = DiffusionGraphConv(joined, self.hidden_dim, max_diffusion_step, bias_start=0.0)

    def forward(self, x: torch.Tensor, state: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        joined = torch.cat([x, state], dim=-1)
        reset, update = torch.sigmoid(self.gates(joined, supports)).chunk(2, dim=-1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * state], dim=-1), supports))
        return update * state + (1.0 - update) * candidate


class OfficialAdaptedDCRNN(nn.Module):
    """Official-code-adapted DCRNN encoder-decoder with curriculum learning."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int = 1,
        rnn_units: int = 256,
        num_rnn_layers: int = 2,
        horizon: int = 24,
        max_diffusion_step: int = 1,
        cl_decay_steps: int = 2000,
        use_curriculum_learning: bool = True,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)
        self.rnn_units = int(rnn_units)
        self.cl_decay_steps = int(cl_decay_steps)
        self.use_curriculum_learning = bool(use_curriculum_learning)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for layer in range(int(num_rnn_layers)):
            enc_in = int(input_dim) if layer == 0 else self.rnn_units
            dec_in = self.output_dim if layer == 0 else self.rnn_units
            self.encoder.append(OfficialDCGRUCell(enc_in, self.rnn_units, max_diffusion_step))
            self.decoder.append(OfficialDCGRUCell(dec_in, self.rnn_units, max_diffusion_step))
        self.output_projection = nn.Linear(self.rnn_units, self.output_dim, bias=False)

    def sampling_threshold(self, batches_seen: int) -> float:
        k = max(float(self.cl_decay_steps), 1.0)
        return float(k / (k + math.exp(float(batches_seen) / k)))

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        labels: torch.Tensor | None = None,
        batches_seen: int = 0,
    ) -> torch.Tensor:
        batch, seq_len, nodes, _ = x.shape
        if nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, received {nodes}.")
        supports = random_walk_supports(adj.to(device=x.device, dtype=x.dtype))
        states = [x.new_zeros(batch, nodes, self.rnn_units) for _ in self.encoder]
        for step in range(seq_len):
            layer_input = x[:, step]
            for layer, cell in enumerate(self.encoder):
                states[layer] = cell(layer_input, states[layer], supports)
                layer_input = states[layer]
        decoder_input = x.new_zeros(batch, nodes, self.output_dim)
        outputs = []
        for step in range(self.horizon):
            layer_input = decoder_input
            for layer, cell in enumerate(self.decoder):
                states[layer] = cell(layer_input, states[layer], supports)
                layer_input = states[layer]
            prediction = self.output_projection(layer_input)
            outputs.append(prediction)
            decoder_input = prediction
            if self.training and labels is not None:
                if self.use_curriculum_learning:
                    threshold = self.sampling_threshold(batches_seen)
                    if torch.rand((), device=x.device).item() < threshold:
                        decoder_input = labels[:, step, :, : self.output_dim]
                else:
                    decoder_input = labels[:, step, :, : self.output_dim]
        return torch.stack(outputs, dim=1)