import unittest

import torch
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.nn import functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ModelNet
from torch_geometric.utils import to_dense_batch

from src.perceiver import Perceiver

class PerceiverModelTest(unittest.TestCase):

    def test_perceiver_forward(self):
        """
        Test the perceiver forward pass

        :return: check if the forward pass is correct
        """

        # [batch_size, length, channels]
        B, M, C = 32, 20, 3
        D, N = 64, 128

        # Create a tensor of shape [batch_size, length, channels]
        x = torch.randn((B, M, C))

        # Create a perceiver model
        perceiver = Perceiver(
            input_dim=C,
            emb_dim=D,
            latent_dim=N,
            num_classes=10,
            depth=2,
            latent_blocks=2,
            heads=4,
            fourier_encode=True,
            max_freq=10,
            num_bands=2
        )
        y = perceiver(x)

        # Return a tensor of shape [Batch, Num_classes]
        self.assertEqual(y.shape, (B, 10))

        # Assert that the values of the input tensor are not all zeros or NaN
        # print(f"Result: {y}")
        self.assertTrue(torch.any(x != 0))
        self.assertFalse(torch.any(torch.isnan(x)))

    def test_logits(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        batch_size = 2  # 64/512 

        pre_transform, transform = T.NormalizeScale(), T.SamplePoints(2000)

        # Training and test datasets
        train_dataset = ModelNet(root="./dataset", name="40", train=True, transform=transform, pre_transform=pre_transform)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        input_dim = 3
        len_shape = 1
        emb_dim = 512  # 512
        latent_dim = 512  # 512
        num_classes = 40

        depth = 2  # 2  
        latent_block = 6 # 6  
        max_freq = 1120  # 1120  
        num_bands = 64  # 64 

        model = Perceiver(
            input_dim=input_dim,
            len_shape=len_shape,
            emb_dim=emb_dim,
            latent_dim=latent_dim,
            num_classes=num_classes,
            depth=depth,
            latent_blocks=latent_block,
            heads=8,
            fourier_encode=True,
            max_freq=max_freq,
            num_bands=num_bands
        ).to(device)

        for (i, batch) in enumerate(tqdm(train_dataloader, desc=f"Training epoch {1}", leave=True)):
            xs, mask = to_dense_batch(batch.pos, batch=batch.batch)
            print(f"xs: {xs.shape} - {xs}, mask: {mask.shape} - {mask}")
            xs, mask = xs.to(device), mask.to(device)
            ys = batch.y

            # Forward pass
            logits = model(xs, mask)

            # Compute the cross entropy loss
            loss = F.cross_entropy(logits, ys)

            break

if __name__ == '__main__':
    unittest.main()