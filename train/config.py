from attr import dataclass

from src.perceiver import Perceiver


@dataclass
class PerceiverCfg:
    """
    Configuration for Perceiver model

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


def get_perceiver_model(cfg: PerceiverCfg):
    """
    Get Perceiver model

    Args:
    cfg: PerceiverCfg: Configuration for Perceiver model

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