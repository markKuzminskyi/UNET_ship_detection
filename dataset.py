import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import yaml


# Downloading config file
config_path = "config.yml"
with open(config_path, "r") as config_file:
    config = yaml.safe_load(config_file)


# Function that decode masks rle
def rle_decode(mask_rle, shape=(768, 768)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    ends = starts + lengths
    im = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        im[lo:hi] = 1
    return im.reshape(shape).T


def get_mask_encodings(masks_data, fnames):
    a = masks_data[masks_data['ImageId'].isin(fnames)]
    return a.groupby('ImageId')['EncodedPixels'].apply(lambda x: x.tolist()).to_dict()


# Creating our image dataset class
class ImgDataset(Dataset):

    def __init__(self,
                 img_dpath,
                 img_fnames,
                 img_transform,
                 mask_encodings,
                 mask_size,
                 mask_transform=None):

        self.img_dpath = img_dpath
        self.img_fnames = img_fnames
        self.img_transform = img_transform

        self.mask_encodings = mask_encodings
        self.mask_size = mask_size
        self.mask_transform = mask_transform

    def __getitem__(self, i):
        seed = np.random.randint(config["random_state"])

        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dpath, fname)
        img = Image.open(fpath)
        if self.img_transform is not None:
            np.random.seed(seed)
            img = self.img_transform(img)

        # Create mask of an image
        mask = np.zeros(self.mask_size, dtype=np.uint8)
        if self.mask_encodings[fname][0] == self.mask_encodings[fname][0]:  # NaN doesn't equal to itself
            for encoding in self.mask_encodings[fname]:
                mask += rle_decode(encoding, self.mask_size)
        mask = np.clip(mask, 0, 1)

        mask = Image.fromarray(mask)

        np.random.seed(seed)
        mask = self.mask_transform(mask)

        return img, torch.from_numpy(np.array(mask, dtype=np.int64))

    def __len__(self):
        return len(self.img_fnames)


# This function returns train and test datasets we work with
def get_data():
    #
    masks_data = pd.read_csv(config["masks_path"])
    masks_data['EncodedPixels_flag'] = masks_data['EncodedPixels'].map(lambda v: 1 if isinstance(v, str) else 0)

    imgs = masks_data.groupby('ImageId').agg({'EncodedPixels_flag': 'sum'}).reset_index().rename(
        columns={'EncodedPixels_flag': 'ships'})

    imgs_w_ships = imgs[imgs['ships'] > 0]
    imgs_wo_ships = imgs[imgs['ships'] == 0].sample(20000, random_state=config["random_state"])

    selected_imgs = pd.concat((imgs_w_ships, imgs_wo_ships))
    selected_imgs['has_ship'] = selected_imgs['ships'] > 0

    train_imgs, val_imgs = train_test_split(selected_imgs, test_size=0.15,
                                            stratify=selected_imgs['has_ship'], random_state=config["random_state"])

    train_fnames = train_imgs['ImageId'].values
    val_fnames = val_imgs['ImageId'].values

    train_transforms = transforms.Compose([transforms.Resize((768, 768)), transforms.ToTensor()])
    val_transforms = transforms.Compose([transforms.Resize((768, 768)), transforms.ToTensor()])
    mask_transforms = transforms.Compose([transforms.Resize((768, 768))])

    train_ds = ImgDataset(config["train_path"], train_fnames, train_transforms,
                          get_mask_encodings(masks_data, train_fnames), config["img_size"], mask_transforms)
    test_ds = ImgDataset(config["train_path"], val_fnames, val_transforms,
                         get_mask_encodings(masks_data, val_fnames), config["img_size"], mask_transforms)

    return train_ds, test_ds
