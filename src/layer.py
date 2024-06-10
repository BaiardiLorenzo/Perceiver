from torch import nn, Tensor


class DenseBlock(nn.Module):
    def __init__(self, emb_dim: int, lin_dim: int = None, dropout: float = 0.0):
        """
        DenseBlock layer:
        - Normalize the input
        - Apply a linear layer
        - Apply GELU activation function
        - Apply a linear layer
        - Apply dropout (if dropout > 0.0)

        In the dense block, inputs are processed with layer norm, 
        passed through a linear layer, activated with a GELU nonlinearity (Hendrycks
        & Gimpel, 2016), and passed through a final linear layer.
        We used dropout throughout the network in earlier experiments,
        but we found this led to degraded performance, so no dropout is used

        :param emb_dim: Embedding dimension
        :param lin_dim: Linear dimension for Linear layer, 
        :param dropout: Dropout rate
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.lin_dim = lin_dim if lin_dim is not None else emb_dim
        self.dropout = dropout

        self.norm = nn.LayerNorm(self.emb_dim)

        self.net = nn.Sequential(
            nn.Linear(self.emb_dim, self.lin_dim),
            nn.GELU(),
            nn.Linear(self.lin_dim, self.emb_dim),
            nn.Dropout(self.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.net(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, emb_dim: int, input_dim: int, heads: int, dropout: float = 0.0):
        """
        TODO Rewrite MultiHeadAttention separately?
        Attention block:
        - Normalize the input
        - Normalize the latent tensor
        - Apply cross attention
        - Apply dense layer

        In the cross-attention module, inputs are first processed with layer norm (Ba et al., 2016)
        before being passed through linear layers to produce each of
        the query, key, and value inputs to the QKV cross-attention
        operation. The queries, keys, and values have the same
        number of channels as the minimum of the input channels,
        which is typically the key/value input (i.e. 261 for ImageNet)
        The output of attention is passed through an additional linear
        layer to project it to the same number of channels in the
        query inputs (so it can be added residually).

        :param emb_dim:
        :param input_dim:
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
            bias=False,
        )

        # Dense layer
        self.dense = DenseBlock(emb_dim, dropout=dropout)

    def forward(self, x: Tensor, z: Tensor, key_mask: Tensor = None) -> Tensor:
        """
        Forward pass

        :param x: input tensor
        :param z: input/latent tensor
        :param key_mask: key padding mask for the unbatched input tensor
        :return:
        """
        # Normalize the input
        x_norm = self.input_norm(x)
        z_norm = self.latent_norm(z)

        # print(x_norm.shape, z_norm.shape)
        # print(key_mask.shape)

        # Compute the cross attention
        a, _ = self.attention(query=z_norm, key=x_norm, value=x_norm, key_padding_mask=key_mask)

        # Add residual connection
        res = a + z

        # Compute dense layer
        out = self.dense(res)

        # Add residual connection
        return out + res


class LatentBlock(nn.Module):
    def __init__(self, emb_dim: int, heads: int, latent_blocks: int, dropout: float = 0.0):
        """
        Latent block

        :param emb_dim:
        :param heads:
        :param latent_blocks:
        :param dropout:
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.heads = heads
        self.latent_blocks = latent_blocks
        self.dropout = dropout

        # Latent transformer
        self.latent_transform = nn.ModuleList([
            AttentionBlock(self.emb_dim, self.emb_dim, self.heads, dropout=self.dropout)
            for _ in range(self.latent_blocks)
        ])

    def forward(self, z: Tensor) -> Tensor:
        """
        Forward pass

        :param z: latent tensor
        :return:
        """
        for latent_transform in self.latent_transform:
            z = latent_transform(z, z)
        return z


class PerceiverBlock(nn.Module):
    def __init__(self, emb_dim: int, input_dim: int, heads: int, latent_blocks: int = 6, dropout: float = 0.0):
        """
        Perceiver block
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.heads = heads 
        self.latent_blocks = latent_blocks
        self.dropout = dropout

        # Cross attention
        self.cross_attention = AttentionBlock(self.emb_dim, self.input_dim, heads=1, dropout=self.dropout)

        # Latent transformer
        self.latent_transform = LatentBlock(self.emb_dim, self.heads, self.latent_blocks, dropout=self.dropout)

    def forward(self, x: Tensor, z: Tensor, key_mask: Tensor = None) -> Tensor:
        """
        Forward pass

        :param x: input tensor
        :param z: latent tensor
        :param mask: key padding mask for the unbatched input tensor
        :return:
        """
        z = self.cross_attention(x, z, key_mask)
        z = self.latent_transform(z)
        return z


class Classifier(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        """
        Classifier

        :param emb_dim:
        :param num_classes:
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes

        # Classifier
        self.fc1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.fc2 = nn.Linear(self.emb_dim, self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass

        :param x:
        :return:
        """
        x = self.fc1(x)
        x = x.mean(dim=0)
        x = self.fc2(x)
        return x
