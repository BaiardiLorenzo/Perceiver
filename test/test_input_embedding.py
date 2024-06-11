import unittest
import torch

from src.embedding import create_latent_array


class InputEmbeddingTest(unittest.TestCase):

    def test_byte_array_shape(self):
        """
        Test that the byte array dimension is correct: 
        [batch_size, height, width, channels] -> [batch_size, height*width, channels]

        :return: check if the shape is correct
        """
        # [batch_size, height, width, channels] -> [batch_size, height*width, channels]
        B, H, W, C = 2, 3, 4, 5
        M = H * W

        # Create a tensor of shape [batch_size, height, width, channels]
        x = torch.zeros((B, H, W, C))

        # Reshape the tensor to [batch_size, height*width, channels]
        x = x.view(B, M, C)

        self.assertEqual(x.shape, (B, M, C))


    def test_latent_array_shape(self):
        """
        Test that the latent array shape is correct:
        [latent_dim, 1, emb_dim]

        :return: check if the shape is correct
        """
        # [latent_dim, 1, emb_dim]
        N, B, D = 5, 1, 4

        # Create a latent array of shape [latent_dim, 1, emb_dim]
        latent = create_latent_array(N, D)  

        self.assertEqual(latent.shape, (N, 1, D))


if __name__ == '__main__':
    unittest.main()
