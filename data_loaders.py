import torchvision
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode


def get_data_loaders(
    dataset_name: str, batch_size: int, is_distributed: bool = False
) -> list[DataLoader]:
    transforms = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.RandomErasing(),
        ]
    )
    if dataset_name == "cifar10":
        datasets = [
            torchvision.datasets.CIFAR10(
                root=".datasets", train=True, transform=transforms, download=True
            ),
            torchvision.datasets.CIFAR10(
                root=".datasets", train=False, transform=T.ToTensor(), download=True
            ),
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if is_distributed:
        data_loaders = [
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=DistributedSampler(dataset),
            )
            for dataset in datasets
        ]
    else:
        data_loaders = [
            DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for dataset in datasets
        ]

    return data_loaders
