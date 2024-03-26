"""
Args:
num_freq_bands: Number of freq bands, with original value (2 * K + 1)
depth: Depth of net.
max_freq: Maximum frequency, hyperparameter depending on how fine the data is.
freq_base: Base for the frequency
input_channels: Number of channels for each token of the input.
input_axis: Number of axes for input data (2 for images, 3 for video)
num_latents: Number of latents, or induced set points, or centroids. Different papers giving it different names.
latent_dim: Latent dimension.
cross_heads: Number of heads for cross attention. Paper said 1.
latent_heads: Number of heads for latent self attention, 8.
cross_dim_head: Number of dimensions per cross attention head.
latent_dim_head: Number of dimensions per latent self attention head.
num_classes: Output number of classes.
attn_dropout: Attention dropout
ff_dropout: Feedforward dropout
weight_tie_layers: Whether to weight tie layers (optional).
fourier_encode_data: Whether to auto-fourier encode the data, using he input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself.
self_per_cross_attn: Number of self attention blocks per cross attn.
final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
num_freq_bands,
depth,
max_freq,
input_channels = 3,
input_axis = 2,
num_latents = 512,
latent_dim = 512,
cross_heads = 1,
latent_heads = 8,
cross_dim_head = 64,
latent_dim_head = 64,
num_classes = 1000,
attn_dropout = 0.,
ff_dropout = 0.,
weight_tie_layers = False,
fourier_encode_data = True,
self_per_cross_attn = 1,
final_classifier_head = True
