import torch
import torch_geometric.transforms as T

from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import BaseTransform
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


class UnitCubeNormalizer(BaseTransform):
    def __call__(self, data: Data) -> Data:
        # Assume that data.pos contains the node positions as a tensor of shape [num_nodes, num_dimensions]
        """
        Normalize a point cloud to fit within a unit cube centered at the origin.

        Args:
            - data (Data): A PyTorch Geometric Data object containing the point cloud in 'pos' attribute.

        Returns:
            - data (Data): The transformed Data object with normalized points in 'pos' attribute.
        """

        if hasattr(data, 'pos'):
            points = data.pos  

            # Compute min and max along each axis
            min_values, _ = torch.min(points, dim=0)
            max_values, _ = torch.max(points, dim=0)

            # Compute center
            center = (min_values + max_values) / 2.0

            # Compute scaling factor
            scale_factor = 1.0 / torch.max(max_values - min_values)

            # Translate and scale points
            translated_points = points - center
            normalized_points = translated_points * scale_factor

            # Update 'pos' attribute in the data object
            data.pos = normalized_points
        
        return data
    

class ZeroCenter(BaseTransform):
    """
    Zero center the point cloud
    
    Args:
        - data (Data): A PyTorch Geometric Data object containing the point cloud in 'pos' attribute.

    Returns:
        - data (Data): The transformed Data object with zero-centered points in 'pos' attribute.
    """
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'pos'):
            points = data.pos  

            # Compute min and max along each axis
            min_values, _ = torch.min(points, dim=0)
            max_values, _ = torch.max(points, dim=0)

            # Compute center
            center = (min_values + max_values) / 2.0

            # Translate and scale points
            points = points - center

            # Update 'pos' attribute in the data object
            data.pos = points
        
        return data
        

def get_modelnet40_loaders(batch_size: int):
    """
    Get ModelNet40 dataset loaders

    The ModelNet40 dataset is a collection of 3D CAD models from 40 categories.
    Each model is represented as a point cloud with about 2000 points.
    The dataset is split into 9,843 training samples and 2,468 testing samples.

    From the paper:
        To apply our model, we first preprocess point clouds by
        zero-centering them. To augment in training we apply ran-
        dom per-point scaling (between 0.99 and 1.01) followed by
        zero-mean and unit-cube normalization. 

    Every point cloud is represented as a torch_geometric.data.Data object with the following attributes:
    - pos: Tensor of shape (N, 3) where N is the number of points and 3 is the number of dimensions
    - y: Integer representing the class of the object
    - batch: Tensor of shape (N,) where N is the number of points. This is used to indicate which point belongs to which object.

    Preprocessing steps:
        - Zero-center the points
        - Randomly scale the points by a factor between 0.9 and 1.1
        - Zero-mean the points
        - Normalize the points to be inside the unit cube

    Args:
        batch_size: int: Batch size

    Returns:
        DataLoader: Training DataLoader
        DataLoader: Testing DataLoader
    """

    n_points = 2048  
    train_transform = T.Compose([
        ZeroCenter(),
        T.RandomScale((0.9, 1.1)),
        T.Center(),
        UnitCubeNormalizer(),
        T.SamplePoints(n_points),
    ])  

    test_transform = T.Compose([
        ZeroCenter(),
        T.Center(),
        UnitCubeNormalizer(),
        T.SamplePoints(n_points),  
    ])

    # Lenght of ModelNet40 dataset is 9,843 for training and 2,468 for testing
    ds_train = ModelNet(root="./dataset/modelnet40", name="40", train=True, transform=train_transform, num_workers=4)
    ds_test = ModelNet(root="./dataset/modelnet40", name="40", train=False, transform=test_transform, num_workers=4)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    return dl_train, dl_test

