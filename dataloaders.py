from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Here we load the mninst dataset into the ./data directory, normalize and prepare dataloaders
# for the training,validation and test splits
class MNISTDataLoader:
    def __init__(self, args, num_workers=8):
        self.batch_size = args.batch_size
        self.val_split = args.val_split
        self.num_workers = num_workers
        self._prepare_data()

    def _prepare_data(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,)
                ),  # Standard MNIST normalization
            ]
        )

        # Download and load the full training dataset
        full_train_dataset = datasets.MNIST(
            root="./data", train=True, transform=transform, download=True
        )

        # Split into train and validation
        val_size = int(self.val_split * len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )

        # Load test dataset
        self.test_dataset = datasets.MNIST(
            root="./data", train=False, transform=transform, download=True
        )

    def get_dataloaders(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader, test_loader


# In a similar way as before, we define a class to load and preprocess the dataset(CIFAR100)
class CIFAR100DataLoader:
    def __init__(self, args, num_workers=8):
        self.batch_size = args.batch_size
        self.val_split = args.val_split
        self.num_workers = num_workers
        self._prepare_data()

    def _prepare_data(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        # Download and load the full training dataset
        full_train_dataset = datasets.CIFAR100(
            root="./data", train=True, transform=transform, download=True
        )

        # Split into train and validation
        val_size = int(self.val_split * len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )

        # Load test dataset
        self.test_dataset = datasets.CIFAR100(
            root="./data", train=False, transform=transform_test, download=True
        )

    def get_dataloaders(self):
        """
        Create and return train, validation, and test data loaders.

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader, test_loader


class CIFAR10DataLoader:
    def __init__(self, args, num_workers=8):
        """
        Initialize CIFAR10 DataLoader.

        Args:
            args: Command line arguments or config object containing:
                - batch_size: Batch size for data loaders
                - val_split: Fraction of training data to use for validation
            num_workers: Number of workers for data loading (default: 8)
        """
        self.batch_size = args.batch_size
        self.val_split = args.val_split
        self.num_workers = num_workers
        self._prepare_data()

    def _prepare_data(self):
        # CIFAR10 normalization values (mean and std for the three channels)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),  # Mean for each channel
                    (0.2470, 0.2435, 0.2616),  # Std for each channel
                ),
            ]
        )

        # Download and load the full training dataset
        full_train_dataset = datasets.CIFAR10(
            root="./data", train=True, transform=transform, download=True
        )

        # Split into train and validation
        val_size = int(self.val_split * len(full_train_dataset))
        train_size = len(full_train_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )

        # Load test dataset
        self.test_dataset = datasets.CIFAR10(
            root="./data", train=False, transform=transform, download=True
        )

    def get_dataloaders(self):
        """
        Create and return train, validation, and test data loaders.

        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader, test_loader
