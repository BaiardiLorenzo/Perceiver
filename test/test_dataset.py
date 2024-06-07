from re import S
import unittest
import torch

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints


class TestModelNet(unittest.TestCase):

    def test_train_dataset(self):
        """
        Test that the training dataset is loaded correctly
        :return:
        """
        train_dataset = ModelNet(root="./dataset", name="40", train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        self.assertEqual(len(train_dataset), 9843)
        self.assertEqual(len(train_dataloader), 154)

    def test_test_dataset(self):
        """
        Test that the test dataset is loaded correctly
        :return:
        """
        test_dataset = ModelNet(root="./dataset", name="40", train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        self.assertEqual(len(test_dataset), 2468)
        self.assertEqual(len(test_dataloader), 39)

    def test_get_item(self):
        """
        Test that the get item method works correctly
        :return:
        """
        sample_points = SamplePoints(1024)
        train_dataset = ModelNet(root="./dataset", name="40", train=True, transform=sample_points)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        for batch in train_dataloader:
            print(batch)
            break


if __name__ == '__main__':
    unittest.main()
