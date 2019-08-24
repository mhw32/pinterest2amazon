import os
from torchvision import datasets, transforms


class BaseDataset(data.Dataset):
    def __init__(self, data_dir, split='train', image_transforms=None):
        super().__init__()
        assert split in ['train', 'validation', 'test']

        if image_transforms is None:
            image_transforms = transforms.ToTensor()

        self.data_dir = os.path.join(data_dir, split)
        self.dataset = datasets.ImageFolder(self.data_dir, image_transforms)

    def __getitem__(self, index):
        image_data = list(self.dataset.__getitem__(index))
        # important to return the index!
        data = [index] + image_data
        return tuple(data)

    def __len__(self):
        return len(self.dataset)
