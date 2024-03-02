from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
import pandas as pd
import os


class VAEFacesDataset(Dataset):
    def __init__(self, image_path, labels_path, transforms=None):
        self.image_path = image_path
        self.labels = pd.read_csv(labels_path, index_col='image_name')
        self.transforms = transforms

    def __getitem__(self, index):
        image_file = os.listdir(self.image_path)[index]
        image = Image.open(os.path.join(self.image_path, image_file))

        if self.transforms:
            image = self.transforms(image)

        label = self.labels.loc[image_file].item()
        return image, label

    def __len__(self):
        return len(os.listdir(self.image_path))
