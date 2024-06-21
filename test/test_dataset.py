from re import S
import unittest
import torch

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch_geometric.utils import to_dense_batch


class TestModelNet(unittest.TestCase):

    def test_train_dataset(self):
        """
        Test that the training dataset is loaded correctly
        :return:
        """
        train_dataset = ModelNet(root="./dataset/modelnet40", name="40", train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        self.assertEqual(len(train_dataset), 9843)
        self.assertEqual(len(train_dataloader), 154)


    def test_test_dataset(self):
        """
        Test that the test dataset is loaded correctly
        :return:
        """
        test_dataset = ModelNet(root="./dataset/modelnet40", name="40", train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

        self.assertEqual(len(test_dataset), 2468)
        self.assertEqual(len(test_dataloader), 39)


    def test_get_dense_batch(self):
        """
        Test that the get item method works correctly
        :return:
        """
        train_dataset = ModelNet(root="./dataset/modelnet40", name="40", train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        for batch in train_dataloader:
            tensor, mask = to_dense_batch(batch.pos, batch=batch.batch)
            print(tensor.shape)
            print(mask.shape)
            break


    def test_sample_points(self):
        """
        Test that the sample points transform works correctly
        :return:
        """
        train_dataset = ModelNet(root="./dataset/modelnet40", name="40", train=True, transform=SamplePoints(1024))
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        for batch in train_dataloader:
            print(batch.pos.shape)
            break

    
    def test_sample_training(self):
        """
        Test that the training dataset is loaded correctly
        """
        train_dataset = ModelNet(root="./dataset/modelnet40", name="40", train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        for (i, batch) in enumerate(tqdm(train_dataloader)):
            print(i, batch.pos.shape, batch.y.shape)
            break

if __name__ == '__main__':
    unittest.main()
