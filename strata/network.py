"""v_theta (paper Sec. 3.4 "network"; Supp. C): pre-norm Transformer, 6 layers, width 256, 4 heads, feed-forward
1024, dropout 0.1, GELU; input linear + sinusoidal positions; each layer adds a projection of the timestep
embedding.  v-parameterisation with the cosine noise schedule (Eq. (5)).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_T = 1000


def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Cumulative signal level of the cosine schedule."""
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    return (f / math.cos(s / (1 + s) * math.pi / 2) ** 2).clamp(1e-5, 1.0)


def alpha_sigma(t_idx: torch.Tensor) -> tuple:
    """(alpha_i, sigma_i) of the cosine noise schedule at step i (Eq. (5))."""
    ab = cosine_alpha_bar((t_idx.float() + 1) / N_T)
    return ab.sqrt(), (1 - ab).sqrt()


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


def r6_orthonormal(r6: torch.Tensor) -> torch.Tensor:
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1e-8)
    return torch.cat([b1, b2], dim=-1)


D = 63


BS, TRANS, ROT, GAZE = slice(0, 52), slice(52, 55), slice(55, 61), slice(61, 63)


N_TEXT = 11


N_LETTERS, LETTER_DIM = 28, 16


N_VISEMES = 15  # 0 = silence, 1..13 visemes, 14 other word frame


L2V2_LEXICAL = {"viseme_v2": N_VISEMES, "letters": N_LETTERS}  # lexical -> embedding rows


class Denoiser(nn.Module):
    def __init__(self, free: np.ndarray, d_model: int = 256, n_layers: int = 6, n_heads: int = 4, ff: int = 1024,
                 dropout: float = 0.1, n_timeline: int = 0, n_v2: int = 0, n_lexical: int = N_LETTERS):
        super().__init__()
        self.d_model = int(d_model)
        self.register_buffer("free", torch.as_tensor(np.asarray(free, dtype=np.float32)))
        self.letters = nn.Embedding(int(n_lexical), LETTER_DIM, padding_idx=0)  # the 15 viseme rows
        self.n_timeline = int(n_timeline)
        self.n_v2 = int(n_v2)
        self.inp = nn.Linear(4 * D + N_TEXT + LETTER_DIM + self.n_timeline + self.n_v2, d_model)
        self.t_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout, activation="gelu",
                                                               batch_first=True, norm_first=True) for _ in range(n_layers))
        self.cond_proj = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(n_layers))
        self.norm_out = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, D)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, t_idx, x_obs, m, rest, text, letters, pad=None, timeline=None, v2=None):
        """v_theta(z_i, i, c(N, w)) (Eq. (5)): z (B, L, 63) noised motion; t_idx (B,) step i; x_obs, m (B, L, 63);
        rest (B, 63); text (B, L, 11) timing features; letters (B, L) viseme ids; pad (B, L) True = real frame;
        v2 (B, L, 40) notes block; timeline unused (None).  -> v (B, L, 63), zero on excluded columns."""
        L = z.shape[1]
        parts = [z, x_obs, m, rest[:, None, :].expand(-1, L, -1), text, self.letters(letters.long())]
        if (timeline is None) != (self.n_timeline == 0):
            raise ValueError("timeline input given to a non-timeline model or missing for a timeline model")
        if timeline is not None:
            parts.append(timeline)
        if (v2 is None) != (self.n_v2 == 0):
            raise ValueError("notes block given to a model without one, or missing")
        if v2 is not None:
            parts.append(v2)
        h = self.inp(torch.cat(parts, dim=-1)) + timestep_embedding(torch.arange(L, device=z.device), self.d_model)[None]
        e = self.t_mlp(timestep_embedding(t_idx, self.d_model))
        mask = None if pad is None else ~pad.bool()
        for layer, proj in zip(self.layers, self.cond_proj):
            h = layer(h + proj(e)[:, None], src_key_padding_mask=mask)
        return self.out(self.norm_out(h)) * self.free


def build_model(c: dict, free: np.ndarray, n_v2: int = None) -> Denoiser:
    """v_theta (Sec. 3.4 "network"): the 6-layer Transformer of a configuration with the notes block of width
    n_v2 (40) and the 15-row viseme embedding."""
    mc = c["model"]
    return Denoiser(free, mc["d_model"], mc["n_layers"], mc["n_heads"], mc["ff"], mc["dropout"], 0, int(n_v2),
                    L2V2_LEXICAL[c["l2v2"]["lexical"]])
