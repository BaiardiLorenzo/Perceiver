from torch import nn, Tensor


class DenseBlock(nn.Module):
    def __init__(self, emb_dim: int, dropout: float=0.0):
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
        :param dropout: Dropout rate (default: 0.0)
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout

        # Normalization before the dense block
        self.norm = nn.LayerNorm(self.emb_dim)

        # Dense block
        self.net = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.GELU(),
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.Dropout(self.dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: 
        - Normalize the input
        - Apply the dense block

        :param x: input tensor [Lenght, Batch, Emb_dim]
        :return: output tensor [Lenght, Batch, Emb_dim]
        """
        x = self.norm(x)
        x = self.net(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, emb_dim: int, input_dim: int, heads: int=1, dropout: float=0.0):
        # FIXME Rewrite MultiHeadAttention separately?

        """
        Attention block:
        - Normalize the inputs (latent and byte tensors)
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

        :param emb_dim: Embedding dimension
        :param input_dim: Input dimension
        :param heads: Number of heads in the multi-head attention (default: 1)
        :param dropout: Dropout rate (default: 0.0)
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
            dropout=self.dropout
        )

        # Dense layer
        self.dense = DenseBlock(emb_dim, dropout=dropout)

    def forward(self, x: Tensor, z: Tensor, key_mask: Tensor=None) -> Tensor:
        """
        Forward pass:
        - Normalize the inputs (latent and byte tensors)
        - Compute the cross attention
        - Add residual connection (before the normalization)
        - Compute dense layer
        - Add residual connection (before the dense layer)

        :param x: input tensor [Lenght, Batch, Input_dim]
        :param z: input/latent tensor [Lenght, Batch, Emb_dim]
        :param key_mask: key padding mask for the unbatched input tensor 
        :return: output tensor [Lenght, Batch, Emb_dim]
        """
        # Normalize the inputs (latent and byte tensors)
        x_norm = self.input_norm(x)
        z_norm = self.latent_norm(z)

        # Compute the cross attention
        a, _ = self.attention(
            query=z_norm, 
            key=x_norm, 
            value=x_norm, 
            key_padding_mask=key_mask
        )

        # Add residual connection
        res = a + z

        # Compute dense layer
        out = self.dense(res)

        # Add residual connection
        return out + res


class LatentTransformerBlock(nn.Module):
    def __init__(self, emb_dim: int, heads: int, latent_blocks: int, dropout: float=0.0):
        """
        Latent Transformer block

        :param emb_dim: Embedding dimension
        :param heads: Number of heads in the multi-head attention
        :param latent_blocks: Number of latent blocks
        :param dropout: Dropout rate (default: 0.0)
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
        Forward pass:
        - Apply the latent transformer block

        :param z: latent tensor
        :return: output tensor
        """
        for lt in self.latent_transform:
            z = lt(z, z)
        return z


class PerceiverBlock(nn.Module):
    def __init__(self, emb_dim: int, input_dim: int, heads: int, latent_blocks: int, dropout: float=0.0):
        """
        Perceiver block:
        - One cross attention block
        - Multiple latent transformer blocks

        :param emb_dim: Embedding dimension
        :param input_dim: Input dimension
        :param heads: Number of heads in the multi-head attention for the transformer block
        :param latent_blocks: Number of latent blocks
        :param dropout: Dropout rate (default: 0.0)
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.heads = heads 
        self.latent_blocks = latent_blocks
        self.dropout = dropout

        # Cross attention
        self.cross_attention = AttentionBlock(
            self.emb_dim, 
            self.input_dim,
            dropout=self.dropout
        )

        # Latent transformer
        self.latent_transform = LatentTransformerBlock(
            self.emb_dim, 
            self.heads, 
            self.latent_blocks, 
            dropout=self.dropout
        )

    def forward(self, x: Tensor, z: Tensor, key_mask: Tensor=None) -> Tensor:
        """
        Forward pass:
        - Apply the cross attention block (with mask if provided)
        - Apply the latent transformer block

        :param x: input tensor [Lenght, Batch, Input_dim]
        :param z: latent tensor [Lenght, Batch, Emb_dim]
        :param mask: key padding mask for the unbatched input tensor
        :return: output tensor [Lenght, Batch, Emb_dim]
        """
        z = self.cross_attention(x, z, key_mask)
        z = self.latent_transform(z)
        return z


class Classifier(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        """
        Classifier:
        - Average over the index dimension
        - Apply a linear layer        

        To produce output logits, we average the output of the final
        latent self-attention module over the index dimension.
        This produces a single global summary vector, which we then
        project to the number of target classes using a single linear layer.

        :param emb_dim: Embedding dimension
        :param num_classes: Number of classes for the classification task
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes

        # Classifier
        self.fc = nn.Linear(self.emb_dim, self.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass:
        - Average over the index dimension
        - Apply a linear layer

        :param x: input tensor [Lenght, Batch, Emb_dim]
        :return: output tensor [Batch, Num_classes]
        """
        x = x.mean(dim=0)
        x = self.fc(x)
        return x
