class LatentArray(nn.Module):
    def __init__(self, channel_dim: int, latent_dim: int):
        """
        Latent array

        :param channel_dim: D
        :param latent_dim: N
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.channel_dim = channel_dim

        # The latent array is randomly initialized using a truncated normal distribution with
        # mean 0, standard deviation 0.02, and truncation bounds [-2, 2].
        self.latent = nn.Parameter(torch.nn.init.trunc_normal_(
            torch.zeros(self.channel_dim, self.latent_dim),
            mean=0,
            std=0.02,
            a=-2, b=2)
        )

    def forward(self) -> Tensor:
        return self.latent


class LatentArray(torch.nn.Parameter):
    def __init__(self, channel_dim: int, latent_dim: int, mean: float = 0, std: float = 0.02, a: float = -2, b: float = 2):
        """
        Latent array
        The latent array is randomly initialized using a truncated normal distribution with
        mean 0, standard deviation 0.02, and truncation bounds [-2, 2].

        :param channel_dim: D
        :param latent_dim: N
        :param mean: default value is 0
        :param std: default value is 0.02
        :param a: lower bound, default value is -2
        :param b: upper bound, default value is 2
        """
        super().__init__()
        self.channel_dim = channel_dim
        self.latent_dim = latent_dim
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b

        self.data = torch.nn.init.trunc_normal_(
            torch.zeros(self.channel_dim, self.latent_dim),
            mean=self.mean,
            std=self.std,
            a=self.a, b=self.b
        )


class LatentArray:

    def __init__(self, channel_dim: int, latent_dim: int, mean: float = 0, std: float = 0.02, a: float = -2, b: float = 2):
        """
        Latent array
        The latent array is randomly initialized using a truncated normal distribution with
        mean 0, standard deviation 0.02, and truncation bounds [-2, 2].

        :param channel_dim: D
        :param latent_dim: N
        :param mean: default value is 0
        :param std: default value is 0.02
        :param a: lower bound, default value is -2
        :param b: upper bound, default value is 2
        """
        super().__init__()
        self.channel_dim = channel_dim
        self.latent_dim = latent_dim
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b

        self.parameter = torch.nn.Parameter(torch.nn.init.trunc_normal_(
            torch.zeros(self.channel_dim, self.latent_dim),
            mean=self.mean,
            std=self.std,
            a=self.a, b=self.b
        ))

    def __getattr__(self, name):
        return getattr(self.parameter, name)

    def __repr__(self):
        return repr(self.parameter)