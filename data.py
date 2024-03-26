from torch.utils.data import DataLoader
from torch_geometric.datasets import ModelNet


def main():
    # Training set
    train_dataset = ModelNet(root="./dataset", name="40", train=True)

    # Test set
    test_dataset = ModelNet(root="./dataset", name="40", train=False)

    # Parameters for the dataloader
    batch_size = 32

    # Dataloader for the training set and test set
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":
    main()
