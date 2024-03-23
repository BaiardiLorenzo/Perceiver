import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np


def fourier_encode(x: Tensor, max_freq: int, num_bands: int = 4) -> Tensor:
    """
    Encodes the input tensor using fourier features

    :param x:
    :param max_freq:
    :param num_bands:
    :return:
    """
    x = x.unsqueeze(-1)  # Add a dimension at the end

    freq_scale = torch.linespace(1., max_freq / 2, num_bands)

    # Compute the fourier features
    x_f = x * freq_scale * np.pi

    x = torch.cat([torch.sin(x_f), torch.cos(x_f)], dim=-1)
    x = torch.cat((x_f, x), dim=-1)
    return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            # nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)


class Normalize(nn.Module):
    def __init__(self, dim: int, module: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.module = module

    def forward(self, x):
        return self.module(self.norm(x))


class Attention(nn.Module):
    def __init__(self, q_dim: int, k_dim: int, v_dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim

        # Q, K, V linear transformations
        self.to_q = nn.Linear(q_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(k_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(v_dim, heads * dim_head, bias=False)

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Attention formula : softmax(QK^T / sqrt(d_k))V
        score = torch.bmm(q, k.transpose(1, 2))
        score /= self.dim_head ** 0.5
        score = F.softmax(score, dim=-1)
        attention = torch.bmm(score, v)
        return attention


class Perceiver(nn.Module):
    def __init__(
            self,
            depth: int,
            max_freq: int,
            num_bands: int,
            input_dim: int,
            latent_dim: int,
            cross_heads: int = 1,
            latent_heads: int = 8,
            cross_dim_head: int = 64,
            latent_dim_head: int = 64,
            num_latents: int = 256,
            num_cross_attn: int = 1,
    ):
        super().__init__()
        self.depth = depth
        self.max_freq = max_freq
        self.num_bands = num_bands
        self.input_dim = input_dim

        self.cross_att = self.cross_attention()
        self.cross_ff = self.cross_ff()
        self.latent_att = self.latent_attention()
        self.latent_ff = self.latent_ff()

        self.n_cross_attn = 2

        self.layers = nn.ModuleList([])

        for i in range(depth):
            self_attns = nn.ModuleList([])

            for block_ind in range(self.n_cross_attn):
                self_attns.append(nn.ModuleList([
                    self.latent_att(key=block_ind),
                    self.latent_ff(key=block_ind)
                ]))

            self.layers.append(nn.ModuleList([self.cross_attn, self.cross_ff, self_attns]))

        # self.classifier = nn.Linear(latent_dim, 10)

    def forward(self, x):
        return

    def cross_attention(self):
        cross_attn = Attention(self.q_dim, self.k_dim, self.v_dim)
        cross_attn_norm = Normalize(self.dim, cross_attn)
        return cross_attn_norm

    def cross_ff(self):
        cross_ff = FeedForward(self.dim, self.hidden_dim)
        cross_ff_norm = Normalize(self.dim, cross_ff)
        return cross_ff_norm

    def latent_attention(self):
        latent_attn = Attention(self.q_dim, self.k_dim, self.v_dim)
        latent_attn_norm = Normalize(self.dim, latent_attn)
        return latent_attn_norm

    def latent_ff(self):
        latent_ff = FeedForward(self.dim, self.hidden_dim)
        latent_ff_norm = Normalize(self.dim, latent_ff)
        return latent_ff_norm
