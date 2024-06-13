from attr import dataclass

from src.perceiver import Perceiver


@dataclass
class PerceiverImageNetCfg:
    """
    Configuration for Perceiver model on ImageNet
    Perceiver models using the LAMB optimizer (You et al., 2020), 
    which was developed for optimizing Transformer-based models. 
    We trained models for 120 epochs with an initial learning rate of 0.004,
    decaying it by a factor of 10 at [84, 102, 114] epochs. The
    best-performing Perceiver we identified on ImageNet at-
    tends to the input image 8 times, each time processing the
    full 50,176-pixel input array using a cross-attend module
    and a latent Transformer with 6 blocks and one cross-attend
    module with a single head per block. We found that sharing
    the initial cross-attention with subsequent cross-attends led
    to instability in training, so we share all cross-attends after
    the first. The dense subblock of each Transformer block
    doesn’t use a bottleneck. We used a latent array with 512 in-
    dices and 1024 channels, and position encodings generated
    with 64 bands and a maximum resolution of 224 pixels. On
    ImageNet, we found that models of this size overfit without
    weight sharing, so we use a model that shares weights for
    all but the first cross-attend and latent Transformer mod-
    ules.

    Attributes:
    input_dim: int: Number of input dimensions
    len_shape: int: Length of the shape
    emb_dim: int: Embedding dimension
    latent_dim: int: Latent dimension
    num_classes: int: Number of classes
    depth: int: Depth of the model
    latent_block: int: Number of latent blocks
    max_freq: int: Maximum frequency
    num_bands: int: Number of bands
    heads: int: Number of heads
    fourier_encode: bool: Whether to use Fourier encoding
    """
    input_dim: int = 3
    len_shape: int = 2
    latent_dim: int = 512
    emb_dim: int = 1024
    num_classes: int = 1000
    depth: int = 1
    latent_blocks: int = 6
    max_freq: int = 224
    num_bands: int = 64
    heads: int = 8
    fourier_encode: bool = True


@dataclass
class PerceiverModelnet40Cfg:
    """
    Configuration for Perceiver model on ModelNet40

    We used an architecture with 2 cross-attentions and 6 self-
    attention layers for each block and otherwise used the same
    architectural settings as ImageNet. We used a higher max-
    imum frequency than for image data to account for the
    irregular sampling structure of point clouds - we used a max
    frequency of 1120 (10×the value used on ImageNet). We
    obtained the best results using 64 frequency bands, and we 
    noticed that values higher than 256 generally led to more
    severe overfitting. We used a batch size of 512 and trained
    with LAMB with a constant learning rate of 1 ×10−3: mod-
    else saturated in performance within 50,000 training steps

    Attributes:
    input_dim: int: Number of input dimensions
    len_shape: int: Length of the shape
    emb_dim: int: Embedding dimension
    latent_dim: int: Latent dimension
    num_classes: int: Number of classes
    depth: int: Depth of the model
    latent_block: int: Number of latent blocks
    max_freq: int: Maximum frequency
    num_bands: int: Number of bands
    heads: int: Number of heads
    fourier_encode: bool: Whether to use Fourier encoding
    """
    input_dim: int = 3
    len_shape: int = 1
    emb_dim: int = 1024
    latent_dim: int = 512
    num_classes: int = 40
    depth: int = 2
    latent_blocks: int = 6
    max_freq: int = 1120
    num_bands: int = 64
    heads: int = 8
    fourier_encode: bool = True


def get_perceiver_model(cfg):
    """
    Get Perceiver model

    Args:
    cfg: Configuration for Perceiver model

    Returns:
    Perceiver: Perceiver model
    """
    return Perceiver(
        input_dim=cfg.input_dim,
        len_shape=cfg.len_shape,
        emb_dim=cfg.emb_dim,
        latent_dim=cfg.latent_dim,
        num_classes=cfg.num_classes,
        depth=cfg.depth,
        latent_blocks=cfg.latent_blocks,
        max_freq=cfg.max_freq,
        num_bands=cfg.num_bands,
        heads=cfg.heads,
        fourier_encode=cfg.fourier_encode
    )