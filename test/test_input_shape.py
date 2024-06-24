import unittest
import torch

from src.latent_array import get_latents_array


class InputShapeTest(unittest.TestCase):

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
        x = x.view(x.shape[0], -1, x.shape[-1])

        self.assertEqual(x.shape, (B, M, C))

    
    def test_byte_array_pc(self):
        """
        Test that the byte array dimension is correct: 
        [batch_size, height, width, channels] -> [batch_size, height*width, channels]

        :return: check if the shape is correct
        """
        # [batch_size, height, width, channels] -> [batch_size, height*width, channels]
        B, L, C = 2, 4, 5

        # Create a tensor of shape [batch_size, height, width, channels]
        x = torch.zeros((B, L, C))
        # print(x.shape)
        
        # Reshape the tensor to [batch_size, height*width, channels]
        x = x.view(x.shape[0], -1, x.shape[-1])
        # print(x.shape)

        self.assertEqual(x.shape, (B, L, C))


    def test_latent_array_shape(self):
        """
        Test that the latent array shape is correct:
        [latent_dim, 1, emb_dim]

        :return: check if the shape is correct
        """
        # [latent_dim, emb_dim]
        N, D = 5, 4

        # Create a latent array of shape [latent_dim, emb_dim]
        latent = get_latents_array(N, D)  
        # print(latent)

        # Repeat the latent tensor to match the batch size
        # [Emb_length, Emb_dim] -> [Batch, Emb_length, Emb_dim]
        latent = latent.repeat(2, 1, 1)
        # print(latent.shape)
        # print(latent)

        self.assertEqual(latent.shape, (2, N, D))


if __name__ == '__main__':
    unittest.main()
