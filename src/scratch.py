
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



