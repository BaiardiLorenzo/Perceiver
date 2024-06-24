import torch
import torch.nn as nn
import math 

from torch import Tensor
from typing import Optional

class MultiHeadAttention(nn.Module):
    """
    From the paper:
        In the cross-attention module the queries, keys, and values have the same
        number of channels as the minimum of the input channels,
        which is typically the key/value input (i.e. 261 for ImageNet)
        The output of attention is passed through an additional linear
        layer to project it to the same number of channels in the
        query inputs (so it can be added residually).
    """

    def __init__(self, query_dim: int, kv_dim: Optional[int] = None, heads: Optional[int] = 1, dropout: Optional[float] = 0.0) -> None:
        """
        Initialize the MultiHeadAttention layer:
            - Linear layers for query, key and value
            - Linear layer for the output
            - Dropout layer

        Args:
            query_dim (int): The query dimension
            kv_dim (int, optional): The key and value dimension. Defaults to None.
            heads (int, optional): The number of heads. Defaults to 1.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
        """

        super().__init__()
        self.query_dim = query_dim 
        self.kv_dim = kv_dim if kv_dim is not None else query_dim
        self.heads = heads 

        assert self.kv_dim % self.heads == 0, f"The linear layer dimension {self.kv_dim} for q, k, v must be divisible by the number of heads {self.heads}"

        self.dk = self.kv_dim // self.heads 

        self.q = nn.Linear(self.query_dim, self.kv_dim) 
        self.k = nn.Linear(self.kv_dim, self.kv_dim) 
        self.v = nn.Linear(self.kv_dim, self.kv_dim)

        self.output = nn.Linear(self.kv_dim, self.query_dim) 

        self.dropout = nn.Dropout(dropout)

    def attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, dropout: nn.Dropout = None) -> Tensor:
        """
        Compute the scaled dot-product attention

        The scaled dot-product attention is calculated as:
        A = softmax(Q * K^T / sqrt(dk)) * V

        Args:
            q (Tensor): The query tensor
            k (Tensor): The key tensor
            v (Tensor): The value tensor
            mask (Optional[Tensor], optional): The mask tensor. Defaults to None.
            dropout (nn.Dropout, optional): The dropout layer. Defaults to None.

        Returns:
            Tensor: The attention tensor
        """

        # [Batch, Heads, Lenght, dk] @ [Batch, Heads, dk, Lenght] --> [Batch, Heads, Lenght, Lenght]
        # Calculate the attention scores: Q * K^T / sqrt(dk)
        a = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        
        if mask is not None:
            # Mask out the attention scores where the mask value is True, with low value (-inf)
            a.masked_fill_(mask, -torch.finfo(a.dtype).max)

        # Apply the softmax function to the attention scores
        a = a.softmax(dim=-1) 

        if dropout is not None:
            a = dropout(a)

        # [Batch, Heads, Lenght, Lenght] @ [Batch, Heads, Lenght, dk] --> [Batch, Heads, Lenght, dk]
        # Calculate the attention values: A * V
        a = torch.matmul(a, v)

        # Combine all the heads together
        # [Batch, Heads, Lenght, dk] --> [Batch, Lenght, Heads, dk] --> [Batch, Lenght, Latent_dim]
        return a.transpose(1, 2).contiguous().view(a.shape[0], -1, self.heads * self.dk)

    def forward(self, q: Tensor, kv: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the multi-head attention

        Args:
            q (Tensor): The query tensor
            kv (Tensor): The key and value tensor
            mask (Optional[Tensor], optional): The mask tensor. Defaults to None.

        Returns:
            Tensor: The output tensor        
        """

        query = self.q(q) # [Batch, Lenght, Query_dim] --> [Batch, Lenght, KV_dim]
        key = self.k(kv) # [Batch, Lenght, KV_dim] --> [Batch, Lenght, KV_dim]
        value = self.v(kv) # [Batch, Lenght, KV_dim] --> [Batch, Lenght, KV_dim]

        batch_size, q_len, _ = query.shape
        _, kv_len, _ = key.shape

        # [Batch, Lenght, KV_dim] --> [Batch, Lenght, Heads, dk] --> [Batch, Heads, Lenght, dk]
        query = query.view(batch_size, q_len, self.heads, self.dk).transpose(1, 2)
        key = key.view(batch_size, kv_len, self.heads, self.dk).transpose(1, 2)
        value = value.view(batch_size, kv_len, self.heads, self.dk).transpose(1, 2)

        # Calculate attention
        a = self.attention(query, key, value, mask, self.dropout)

        # [Batch, Lenght, Latent_dim] --> [Batch, Lenght, Latent_dim]
        return self.output(a)


class DenseBlock(nn.Module):
    """
    From the paper:
        In the dense block, inputs are processed with layer norm, 
        passed through a linear layer, activated with a GELU nonlinearity (Hendrycks
        & Gimpel, 2016), and passed through a final linear layer.
        We used dropout throughout the network in earlier experiments,
        but we found this led to degraded performance, so no dropout is used.
    """
     
    def __init__(self, latent_dim: int, dropout: Optional[float]= 0):
        """
        Initialize the DenseBlock layer

        Args:
            latent_dim (int): The latent dimension
            dropout (float, optional): The dropout rate. Defaults to 0
        """
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass:
            - Normalize the input
            - Apply a linear layer keeping the same dimension
            - Apply GELU activation function
            - Apply a linear layer keeping the same dimension
            - Apply dropout (if dropout > 0)

        Args:
            x (Tensor): input tensor [Batch, Lenght, Latent_dim]

        Returns:
            Tensor: output tensor [Batch, Lenght, Latent_dim]
        """
        return self.net(self.norm(x))


class CrossAttention(nn.Module):
    """
    From the paper:
        In the cross-attention module, inputs are first processed with layer norm (Ba et al., 2016)
        before being passed through linear layers to produce each of
        the query, key, and value inputs to the QKV cross-attention
        operation. The queries, keys, and values have the same
        number of channels as the minimum of the input channels,
        which is typically the key/value input (i.e. 261 for ImageNet)
        The output of attention is passed through an additional linear
        layer to project it to the same number of channels in the
        query inputs (so it can be added residually).
    """
    def __init__(self, latent_dim: int, input_dim: int, dropout: Optional[float] = 0):
        """
        Initialize the CrossAttention module

        Args:
            latent_dim (int): The latent dimension
            input_dim (int): The input dimension
            heads (int, optional): The number of heads. Defaults to 1.
            dropout (float, optional): The dropout rate. Defaults to 0.
        """
        super().__init__()

        # Normalization before the cross attention
        self.q_norm = nn.LayerNorm(latent_dim)
        self.kv_norm = nn.LayerNorm(input_dim)

        # Cross attention
        self.mha = MultiHeadAttention(latent_dim, input_dim, dropout=dropout)

        # Dense layer
        self.dense = DenseBlock(latent_dim, dropout=dropout)

    def forward(self, x: Tensor, z: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass:
            - Normalize the inputs (latent and byte tensors)
            - Compute the cross attention
            - Add residual connection (latent tensor)
            - Compute dense layer
            - Add residual connection (attention tensor)

        Args:
            x (Tensor): input tensor [Batch, Lenght, Input_dim]
            z (Tensor): latent tensor [Batch, Lenght, Latent_dim]
            mask (Tensor, optional): mask tensor. Defaults to None.

        Returns:
            Tensor: output tensor [Batch, Lenght, Latent_dim]
        """
        # Normalize the inputs 
        q_norm = self.q_norm(z)
        kv_norm = self.kv_norm(x)

        # Compute the cross attention
        a = self.mha(q_norm, kv_norm, mask = mask)

        # Add residual connection
        x = a + z

        # Compute dense layer
        return x + self.dense(x)


class SelfAttention(nn.Module):
    """
    From the paper:
        In the self-attention block, inputs are processed with layer
        norm and passed through query, key, and value layers before
        being used to compute QKV self-attention. The output is
        passed through another linear layer.
    """
    def __init__(self, latent_dim: int, heads: int, dropout: float=0.0):
        """
        Initialize the SelfAttention module
        
        Args:
            latent_dim (int): The latent dimension
            heads (int): The number of heads
            dropout (float, optional): The dropout rate. Defaults to 0.0.
        """
        super().__init__()

        # Normalization before the self attention
        self.norm = nn.LayerNorm(latent_dim)

        # Self attention
        self.mha = MultiHeadAttention(latent_dim, heads=heads, dropout=dropout)

        # Dense layer
        self.dense = DenseBlock(latent_dim, dropout=dropout)

    def forward(self, z: Tensor) -> Tensor:
        """
        Forward pass:
            - Normalize the input
            - Compute the self attention
            - Add residual connection (latent tensor)
            - Compute dense layer
            - Add residual connection (attention tensor)

        Args:
            z (Tensor): input tensor [Batch, Lenght, Latent_dim]

        Returns:
            Tensor: output tensor [Batch, Lenght, Latent_dim]
        """
        # Normalize the input
        qkv = self.norm(z)

        # Compute the self attention
        a = self.mha(qkv, qkv)

        # Add residual connection
        x = a + z

        # Compute dense layer
        return x + self.dense(x)


class LatentTransformer(nn.Module):
    def __init__(self, latent_dim: int, heads: int, latent_blocks: int, dropout: Optional[float]= 0):
        """
        Initialize the LatentTransformer module

        Args:
            latent_dim (int): The latent dimension
            heads (int): The number of heads
            latent_blocks (int): The number of latent blocks
            dropout (float, optional): The dropout rate. Defaults to 0.
        """
        super().__init__()

        # Latent transformer
        self.layers = nn.ModuleList([SelfAttention(latent_dim, heads, dropout) for _ in range(latent_blocks)])

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass:
            - Apply the latent transformer block

        Args:
            x (Tensor): latent tensor
        """
        for layer in self.layers:
            x = layer(x)
        return x


class PerceiverBlock(nn.Module):
    def __init__(self, latent_dim: int, input_dim: int, latent_blocks: int, heads: int, dropout: Optional[float] = 0):
        """
        Initialize the PerceiverBlock module:
            - Cross attention
            - Latent transformer

        Args:
            latent_dim (int): The latent dimension
            input_dim (int): The input dimension
            heads (int): The number of heads
            latent_blocks (int): The number of latent blocks
            dropout (float, optional): The dropout rate. Defaults to 0.0.
        """
        super().__init__()

        # Cross attention
        self.cross_attention = CrossAttention(latent_dim, input_dim, dropout=dropout)

        # Latent transformer
        self.latent_transform = LatentTransformer(latent_dim, heads, latent_blocks, dropout)

    def forward(self, x: Tensor, z: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass:
            - Apply the cross attention block (with mask if provided)
            - Apply the latent transformer block

        Args:
            x (Tensor): input tensor
            z (Tensor): latent tensor
            mask (Tensor, optional): mask tensor. Defaults to None.
        """
        z = self.cross_attention(x, z, mask)
        return self.latent_transform(z)


class Decoder(nn.Module):
    """
    From the paper:
        To produce output logits, we average the output of the final
        latent self-attention module over the index dimension.
        This produces a single global summary vector, which we then
        project to the number of target classes using a single linear layer.
    """
    def __init__(self, latent_dim: int, num_classes: int):
        """
        Initialize the Decoder module:
            - Average over the index dimension
            - Apply a linear layer        

        Args:
            latent_dim (int): The latent dimension
            num_classes (int): The number of classes
        """
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass:
            - Average over the index dimension
            - Apply the linear layer

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        return self.fc(x.mean(dim=1))
