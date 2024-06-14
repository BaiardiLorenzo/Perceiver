from cgi import test
import torchvision
import torch_geometric.transforms as T

from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import BaseTransform
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

class UnitCubeNormalizer(BaseTransform):
    def __call__(self, data: Data) -> Data:
        # Assume that data.pos contains the node positions as a tensor of shape [num_nodes, num_dimensions]
        if hasattr(data, 'pos'):
            # Find the minimum and maximum position in each dimension
            min_pos = data.pos.min(dim=0, keepdim=True)[0]
            max_pos = data.pos.max(dim=0, keepdim=True)[0]
            
            # Calculate the range of positions
            range_pos = max_pos - min_pos
            
            # Avoid division by zero
            range_pos[range_pos == 0] = 1
            
            # Normalize the positions to the range [0, 1]
            data.pos = (data.pos - min_pos) / range_pos
        
        return data
        

def get_modelnet40_loaders(batch_size: int):
    """
    Get ModelNet40 dataset loaders

    The ModelNet40 dataset is a collection of 3D CAD models from 40 categories.
    Each model is represented as a point cloud with about 2000 points.
    The dataset is split into 9,843 training samples and 2,468 testing samples.

    To apply our model, we first preprocess point clouds by
    zero-centering them. To augment in training we apply ran-
    dom per-point scaling (between 0.99 and 1.01) followed by
    zero-mean and unit-cube normalization. We also explored
    random per-point translation (between -0.02 and 0.02) and
    random point-cloud rotation, but we found this did not im-
    prove performance.

    Every point cloud is represented as a torch_geometric.data.Data object with the following attributes:
    - pos: Tensor of shape (N, 3) where N is the number of points and 3 is the number of dimensions
    - y: Integer representing the class of the object
    - batch: Tensor of shape (N,) where N is the number of points. This is used to indicate which point belongs to which object.

    Args:
    batch_size: int: Batch size

    Returns:
    DataLoader: Training DataLoader
    DataLoader: Validation DataLoader
    DataLoader: Testing DataLoader
    """

    # ZERO-CENTERING THE POINTS: Training and test data should be zero-centered
    # RANDOM SCALING: Randomly scale the points by a factor between 0.9 and 1.1
    # ZERO-MEAN: 
    # UNIT-CUBE NORMALIZATION: Normalize the points to be inside the unit cube

    # FIXME if this transforms are correctly implemented
    # FIXME Best results are obtained with 2000 points and Normalizing Scale
    # TODO without sample points

    n_points = 2048  # Change this to 2048
    train_transform = T.Compose([
        T.SamplePoints(n_points),
        # T.RandomScale((0.9, 1.1)),
        T.NormalizeScale(),
        # T.Center(),
        # T.RandomScale((0.9, 1.1)),
        # T.Center(),
        # UnitCubeNormalizer(),
    ])  

    test_transform = T.Compose([
        T.SamplePoints(n_points),
        T.NormalizeScale(),
        # T.Center(),
        # UnitCubeNormalizer(),
    ])

    # Lenght of ModelNet40 dataset is 9,843 for training and 2,468 for testing
    ds_train = ModelNet(root="./dataset/modelnet40", name="40", train=True, transform=train_transform)
    ds_test = ModelNet(root="./dataset/modelnet40", name="40", train=False, transform=test_transform)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    return dl_train, dl_test


def get_imagenet_loaders(batch_size: int):
    """
    We train our model using images sampled by Inception-style
    preprocessing (Szegedy et al., 2015), including standard
    224 ×224 pixel crops. Additionally, we augment all images
    using RandAugment (Cubuk et al., 2020) at training time.
    """
        
    # RANDOM CROP: Randomly crop the image to 224x224
    # RANDAugment: Apply RandAugment with N=2, M=10 during training

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandAugment(2, 10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ds_train = torchvision.datasets.ImageNet(root="./dataset/imagenet", split="train", transform=train_transform)
    ds_test = torchvision.datasets.ImageNet(root="./dataset/imagenet", split="val", transform=test_transform)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)  

    return dl_train, dl_test


def test_modelnet40():
    # Load the ModelNet40 dataset
    ds_train, ds_test = get_modelnet40_loaders(batch_size=1)

    # Calculate the average length of all point clouds in the training set and test set
    lengths = []

    for data in ds_train:
        lengths.append(data.pos.shape[0])
    
    for data in ds_test:
        lengths.append(data.pos.shape[0])

    avg_length = sum(lengths) / len(lengths)

    print(f"Average length of point clouds: {avg_length}")


def test_imagenet():
    # Load the ImageNet dataset
    ds_train, ds_test = get_imagenet_loaders(batch_size=1)

    # Get the first batch of training data
    data = next(iter(ds_train))
    print(data[0].shape)  # Shape of the image
    print(data[1])  # Class label


if __name__ == "__main__":
    test_imagenet()
