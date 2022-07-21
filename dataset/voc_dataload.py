import numpy as np
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision.transforms import *
from pandas import read_csv
from PIL import Image
from utils.file_utils.bbox2yolov1 import convert_bbox2labels


class VOCDataset(Dataset):
    def __init__(self, image_dir, csv_path, resize):
        super(VOCDataset, self).__init__()
        self.image_dir = image_dir
        self.resize = resize
        self.df = read_csv(csv_path,encoding='utf-8', engine='python')
        self.transformer = Compose([
            Resize(self.resize),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.transformer(Image.open(self.df['img_path'][idx]))
        label = np.load(self.df['object_path'][idx])
        label = convert_bbox2labels(label)

        return image, label


def get_dataloader(image_dir, csv_path, resize, batch_size, num_workders, train_percent=0.9):
    dataset = VOCDataset(image_dir, csv_path, resize)
    num_sample = len(dataset)
    num_train = int(train_percent * num_sample)
    num_valid = num_sample - num_train
    train_ds, valid_ds = random_split(dataset, [num_train, num_valid])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workders, pin_memory=True,
                          persistent_workers=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workders, pin_memory=True,
                          persistent_workers=True)
    return train_dl, valid_dl, len(dataset), len(train_ds), len(valid_ds)

