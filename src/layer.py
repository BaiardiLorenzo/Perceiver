from torch import nn, Tensor


class DenseBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        """
        FeedForward block

        :param dim:
        :param dropout:
        """
        super().__init__()
        self.dim = dim
        self.dropout = dropout

        self.net = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim, self.dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, emb_dim: int, input_dim: int, heads: int, dropout: float = 0.0):
        """
        Attention block
        @TODO Rewrite MultiHeadAttention separately

        :param emb_dim:
        :param heads:
        :param dropout:
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.num_heads = heads
        self.dropout = dropout

        # Normalization before the cross attention
        self.latent_norm = nn.LayerNorm(self.emb_dim)
        self.input_norm = nn.LayerNorm(self.input_dim)

        # Cross attention
        self.attention = nn.MultiheadAttention(
            self.emb_dim,
            self.num_heads,
            kdim=self.input_dim,
            vdim=self.input_dim,
            dropout=self.dropout,
            bias=False
        )

        # Project the output of the cross attention
        self.proj = nn.Linear(emb_dim, emb_dim)

        # Dense layer
        self.dense = DenseBlock(emb_dim, dropout=dropout)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        """
        Forward pass

        :param x: input tensor
        :param z: input/latent tensor
        :return:
        """
        # Normalize the input
        x_norm = self.input_norm(x)
        z_norm = self.latent_norm(z)

        # Compute the cross attention
        a, _ = self.attention(z_norm, x_norm, x_norm)

        # Project the output of the cross attention
        a = self.proj(a)

        # Compute dense layer
        a = self.dense(a)

        return a + z


class PerceiverBlock(nn.Module):
    def __init__(self, dim: int, heads: int, latent_blocks: int = 6, dropout: float = 0.0):
        """
        Perceiver block
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.latent_blocks = latent_blocks
        self.dropout = dropout

        # Cross attention
        self.cross_attentions = AttentionBlock(self.dim, 1, dropout=self.dropout)

        # Latent transformer
        self.latent_transform = nn.ModuleList([
            AttentionBlock(self.dim, self.heads, dropout=self.dropout)
            for _ in range(self.latent_blocks)
        ])

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        """
        Forward pass

        :param x: input tensor
        :param z: latent tensor
        :return:
        """
        z = self.cross_attention(x, z)
        for latent_transform in self.latent_transform:
            z = latent_transform(z, z)
        return z


class Classifier(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        """
        Classifier

        :param dim:
        :param num_classes:
        """
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes

        # Classifier
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        :param x:
        :return:
        """
        return self.classifier(x)
