import unittest
import torch

from math import pi

from src.positional_encoding import ff_positional_encoding, fourier_features

class PositionalEmbeddingTest(unittest.TestCase):

    def test_list_positions(self):
        """
        Test to create a list of positions for each axis
        
        :return: List of positions for each axis
        """
        dims = (4, 5)
        
        xd = torch.stack(list(torch.meshgrid(
        *(torch.linspace(-1, 1, steps=dim) for dim in dims),
        indexing="ij",
        )), dim=-1)
        print(xd)
        print(xd.shape)

        # Assert that the shape of the positions is correct: [[Dims], len(Dims)]
        self.assertEqual(xd.shape, torch.Size([4, 5, 2]))

    def test_create_fourier_encode(self):
        """
        Test to create the fourier encoding of the input tensor

        :return: Add the fourier encoding to the input tensor
        """

        dims = [4, 5]
        max_freq = 1120
        num_bands = 64

        # Create a list of positions for each axis
        xd = torch.stack(list(torch.meshgrid(
            *(torch.linspace(-1, 1, steps=dim) for dim in dims),
            indexing="ij",
            )), dim=-1)

        # Add the fourier channel
        x = xd.unsqueeze(-1) 
        xx = x

        # Create the sequenze of frequencies equally spaced from 1 to max_freq / 2 : [num_bands]
        freqs = torch.linspace(1, max_freq / 2, num_bands)

        # Expand the freqs tensor to match the shape of the input tensor : [[1]*len(x.shape), num_bands]
        freqs = freqs.view((1,) * (len(x.shape) - 1) + (-1,))

        # Compute the fourier features
        x = x * freqs * pi

        # Concatenate the fourier features
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = torch.cat((x, xx), dim=-1)

        # Assert that the shape of the encoded positions is correct: [[Dims], len(Dims), (num_bands * 2 + 1)]
        self.assertEqual(x.shape, torch.Size(dims + [len(dims)] + [num_bands * 2 + 1]))


    def test_create_positional_embedding(self):
        """
        Test to create the positional embedding of the input tensor using fourier encoding
        
        :return: Add the positional embedding to the input tensor
        """
        max_freq = 1120
        num_bands = 64

        B, H, W, C = 32, 4, 5, 3

        # Create a tensor of shape [batch_size, height, width, channels]
        x = torch.randn((B, H, W, C))

        # Get dimensions of the tensor
        b, *dims, c, dtype, device = *x.shape, x.dtype, x.device 

        # Assert that the dimensions are correct
        self.assertEqual(dims, [H, W])

        # Create a list of positions for each axis
        xd = torch.stack(list(torch.meshgrid(
            *(torch.linspace(-1, 1, steps=dim, dtype=dtype, device=device) for dim in dims),
            indexing="ij",
            )), dim=-1)

        # Assert that the shape of the positions is correct: [[Dims], len(Dims)]
        self.assertEqual(xd.shape, torch.Size(dims + [len(dims)]))
        
        # Encode the positions with a fourier feature encoding
        enc_pos = fourier_features(xd, max_freq, num_bands)

        # Assert that the shape of the encoded positions is correct: [[Dims], len(Dims), num_bands * 2 + 1]
        self.assertEqual(enc_pos.shape, torch.Size(dims + [len(dims)] + [num_bands * 2 + 1]))
                         
        # Expand the encoded positions to match the shape of the input tensor: 
        # [[Dims], len(Dims)*(num_bands * 2 + 1)]
        enc_pos = enc_pos.view(*enc_pos.shape[:(len(dims))], -1)
        # [Batch, [Dims], len(Dims)*(num_bands * 2 + 1)]
        enc_pos = enc_pos.unsqueeze(0).expand(b, *enc_pos.shape)

        # Assert that the shape of the expanded encoded positions is correct: [Batch, [Dims], len(Dims)*(num_bands * 2 + 1)]
        self.assertEqual(enc_pos.shape, torch.Size([b] + dims + [len(dims) * (num_bands * 2 + 1)]))

        # Concatenate the encoded positions to the input tensor
        x = torch.cat((x, enc_pos), dim=-1)

        # Assert that the shape of the input tensor is correct: [Batch, [Dims], len(Dims)*(num_bands * 2 + 1) + Channels]
        self.assertEqual(x.shape, torch.Size([b] + dims + [len(dims) * (num_bands * 2 + 1) + c]))

    
    def test_positional_embedding(self):
        """
        Test the positional embedding function

        :return: Tensor with positional embedding
        """
        
        B, H, W, C = 32, 4, 5, 3
        max_freq = 1120
        num_bands = 64

        # Create a tensor of shape [batch_size, height, width, channels]
        x = torch.randn((B, H, W, C))

        # Get dimensions of the tensor
        b, *dims, c = x.shape

        # Add the positional embedding to the input tensor
        x = ff_positional_encoding(x, max_freq, num_bands)

        # Assert that the shape of the input tensor is correct: [Batch, [Dims], len(Dims)*(num_bands * 2 + 1) + Channels]
        self.assertEqual(x.shape, torch.Size([b] + dims + [len(dims) * (num_bands * 2 + 1) + c]))
        
        # Flatten the input tensor
        x = x.view(x.shape[0], -1, x.shape[-1])

        # Assert that the values of the input tensor are not all zeros or NaN
        # print(f"Result of positional embedding: {x}")
        self.assertTrue(torch.any(x != 0))
        self.assertFalse(torch.any(torch.isnan(x)))

    
    def test_ff_formula(self):
        """
        Test the formula for the Fourier features

        :return: Check if the formula is correct
        """
        input_shapes = 1
        num_bands = 64
        input_dim = 3

        # FORMULA: results 132 
        input_dim = (input_shapes * (num_bands * 2 + 1)) + input_dim

        self.assertEqual(input_dim, 132)
                         

if __name__ == '__main__':
    unittest.main()