import torch
import wandb
import os
from glob import glob
from tqdm import tqdm
import torch_geometric.transforms as T

from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import BaseTransform
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


class UnitCubeNormalization(BaseTransform):
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
            center = (min_values + max_values) * 0.5   

            # Compute scaling factor for each axis
            sizes = (max_values - min_values)
            scale_factor = 1/torch.max(sizes)

            # Translate and scale points
            translated_points = points - center
            normalized_points = translated_points * scale_factor

            # Update 'pos' attribute in the data object
            data.pos = normalized_points
        
        return data
    

def get_modelnet40_loaders(batch_size: int):
    """
    Get ModelNet40 dataset loaders

    The ModelNet40 dataset is a collection of 3D CAD models from 40 categories.
    Each model is represented as a point cloud with 2048 points.
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
        - Randomly scale the points by a factor between 0.99 and 1.01
        - Zero-mean the points
        - Normalize the points to be inside the unit cube

    Args:
        batch_size: int: Batch size

    Returns:
        DataLoader: Training DataLoader
        DataLoader: Testing DataLoader
    """

    """
    paper_transforms = T.Compose([
        T.SamplePoints(2048),
        T.Center(),
        T.RandomScale((0.99, 1.01)),
        T.Center(),
        UnitCubeNormalizer(),
    ])
    """
    class_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']

    n_points = 2048  
    train_transform = T.Compose([
        T.SamplePoints(n_points),
        T.Center(),
        T.RandomScale((0.99, 1.01)),
        T.NormalizeScale(),
    ])  

    test_transform = T.Compose([
        T.SamplePoints(n_points),
        T.NormalizeScale(),
    ])

    # Lenght of ModelNet40 dataset is 9,843 for training and 2,468 for testing
    ds_train = ModelNet(root="./dataset/modelnet40", name="40", train=True, transform=train_transform)
    ds_test = ModelNet(root="./dataset/modelnet40", name="40", train=False, transform=test_transform)

    # DataLoader
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    return dl_train, dl_test, train_transform, class_names


def test_unit_cube_normalizer():
    # Test UnitCubeNormalizer
    data = Data(
        pos=torch.rand((100, 3)),
        )

    normalizer = UnitCubeNormalization()
    normalized_data = normalizer(data)

    print(normalized_data.pos)


def visualize_modelnet40():
    wandb_project = "ModelNet40_PointCloud" #@param {"type": "string"}
    wandb_run_name = "ModelNet40_NormalizeScale" #@param {"type": "string"}

    wandb.init(project=wandb_project, name=wandb_run_name, job_type="eda")

    # Set experiment configs to be synced with wandb
    config = wandb.config
    config.display_sample = 2048  #@param {type:"slider", min:256, max:4096, step:16}
    config.modelnet_dataset_alias = "./dataset/modelnet40" #@param ["ModelNet10", "ModelNet40"] {type:"raw"}

    # Classes for ModelNet10 and ModelNet40
    categories = sorted([
        x.split(os.sep)[-2]
        for x in glob(os.path.join(
            config.modelnet_dataset_alias, "raw", '*', ''
        ))
    ])

    config.categories = categories

    transform = T.Compose([
        T.SamplePoints(config.display_sample),
        T.Center(),
        T.RandomScale((0.9, 1.1)),
        T.NormalizeScale(),
    ])
    
    train_dataset = ModelNet(
        root=config.modelnet_dataset_alias,
        name=config.modelnet_dataset_alias[-2:],
        train=True,
        transform=transform,
    )

    val_dataset = ModelNet(
        root=config.modelnet_dataset_alias,
        name=config.modelnet_dataset_alias[-2:],
        train=False,
        transform=transform,
    )

    table = wandb.Table(columns=["Model", "Class", "Split"])
    category_dict = {key: 0 for key in config.categories}
    for idx in tqdm(range(len(train_dataset[:50]))):
        point_cloud = wandb.Object3D(train_dataset[idx].pos.numpy())
        category = config.categories[int(train_dataset[idx].y.item())]
        category_dict[category] += 1
        table.add_data(
            point_cloud,
            category,
            "Train"
        )

    data = [[key, category_dict[key]] for key in config.categories]
    wandb.log({
        f"{config.modelnet_dataset_alias} Class-Frequency Distribution" : wandb.plot.bar(
            wandb.Table(data=data, columns = ["Class", "Frequency"]),
            "Class", "Frequency",
            title=f"{config.modelnet_dataset_alias} Class-Frequency Distribution"
        )
    })

    table = wandb.Table(columns=["Model", "Class", "Split"])
    category_dict = {key: 0 for key in config.categories}
    for idx in tqdm(range(len(val_dataset[:50]))):
        point_cloud = wandb.Object3D(val_dataset[idx].pos.numpy())
        category = config.categories[int(val_dataset[idx].y.item())]
        category_dict[category] += 1
        table.add_data(
            point_cloud,
            category,
            "Test"
        )
    wandb.log({config.modelnet_dataset_alias: table})

    data = [[key, category_dict[key]] for key in config.categories]
    wandb.log({
        f"{config.modelnet_dataset_alias} Class-Frequency Distribution" : wandb.plot.bar(
            wandb.Table(data=data, columns = ["Class", "Frequency"]),
            "Class", "Frequency",
            title=f"{config.modelnet_dataset_alias} Class-Frequency Distribution"
        )
    })

    wandb.finish()


if __name__ == "__main__":
    visualize_modelnet40()
    # test_unit_cube_normalizer()