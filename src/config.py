from attr import dataclass

from src.perceiver import Perceiver


@dataclass
class PerceiverModelNet40Cfg():
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

    Args: 
        input_dim: int: Number of input dimensions
        latent_length: int: Length of the latent array
        latent_dim: int: Dimension of the latent array
        num_classes: int: Number of classes
        latent_blocks: int: Number of latent blocks for the latent transformer
        heads: int: Number of heads for the self-attention layer
        perceiver_block: int: Number of perceiver blocks
        share_weights: bool: Whether to share weights between the perceiver blocks
        ff_pos_encoding: bool: Whether to use Fourier encoding
        input_shapes: int: Number of input shapes
        max_freq: int: Maximum frequency
        num_bands: int: Number of bands
    """
    input_dim: int = 3
    latent_length: int = 512
    latent_dim: int = 1024
    num_classes: int = 40
    latent_blocks: int = 6
    heads: int = 8
    perceiver_block: int = 2
    share_weights: bool = False
    ff_pos_encoding: bool = True
    input_shapes: int = 1
    max_freq: int = 1120
    num_bands: int = 64


def get_perceiver_model(cfg, device):
    """
    Get Perceiver model with the given configuration

    Args:
        cfg: Configuration class for the Perceiver model
        device: str: Device to use for the model

    Returns:
        Perceiver: Perceiver model 
        str: Configuration string
        cfg: Configuration class
    """
    return Perceiver(
        input_dim=cfg.input_dim,
        latent_length=cfg.latent_length,
        latent_dim=cfg.latent_dim,
        num_classes=cfg.num_classes,
        latent_blocks=cfg.latent_blocks,
        heads=cfg.heads,
        perceiver_block=cfg.perceiver_block,
        share_weights=cfg.share_weights,
        ff_pos_encoding=cfg.ff_pos_encoding,
        input_shapes=cfg.input_shapes,
        max_freq=cfg.max_freq,
        num_bands=cfg.num_bands,
    ).to(device), cfg
