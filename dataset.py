import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torchvision.transforms as transforms
import random

from typing import Any
from torch.utils.data import Dataset
from utils import pad_crop, read_envi_file, find_arrays_with_object


class SatelliteDataset(Dataset):
    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        transform : bool = True,
    ) -> None:
        if not os.path.exists(data_dir):
            raise ValueError(f'Provided data_dir: "{data_dir}" does not exist.')
        
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "Image")
        self.mask_dir = os.path.join(data_dir, "Mask")
        self.image_list = pad_crop(read_envi_file(self.image_dir, True), 224)
        self.mask_list = pad_crop(read_envi_file(self.mask_dir, True), 224)
        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.transform = transform

        random.seed(99)
        self.custom_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomRotation(10),
        ])

        #num_samples = len(self.image_list)
        #indices = list(range(num_samples))
        indices = find_arrays_with_object(self.mask_list)
        num_samples = len(indices)

        # Data Split
        np.random.seed(99)
        np.random.shuffle(indices)
        num_val_samples = int(self.val_ratio * num_samples)
        num_test_sampels = int(self.test_ratio * num_samples)
        if self.split == "train":
            self.indices = indices[:-num_val_samples]
        elif self.split == "val":
            self.indices = indices[-num_val_samples:-num_test_sampels]
        elif self.split == 'test':
            self.indices = indices[-num_test_sampels:]
        else:
            raise ValueError("Invalid split value. Use 'train', 'val' or 'test'.")


    def __len__(self) -> int:
        return len(self.indices)


    def __getitem__(self, idx: Any) -> Any:
        img_idx = self.indices[idx]
        img = self.image_list[img_idx]
        mask = self.mask_list[img_idx]
        
        image_np = np.array(img, dtype=np.float32)
        mask_np = np.array(mask, dtype=np.float32)
        
        padded_img = np.pad(image_np, ((0,0),(16,16),(16,16)), 'constant', constant_values=0).swapaxes(0,2)
        padded_mask = np.pad(mask_np, ((0,0),(16,16),(16,16)), 'constant', constant_values=0).swapaxes(0,2)
        
        if self.transform == True :
            processed_img = self.custom_transform(padded_img)
            processed_mask = self.custom_transform(padded_mask)
        else:
            processed_img = transforms.ToTensor(padded_img)
            processed_mask = transforms.ToTensor(padded_mask)

        return processed_img, processed_mask



""" if __name__ == "__main__":
    data_dir = "./data"
    input_size = (256, 256)

    train_transform = A.Compose(
        [
            A.Resize(input_size[0], input_size[1]),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(input_size[0], input_size[1]),
            ToTensorV2(),
        ]
    )

    # Train dataset with defining split
    #train_dataset = CustomDataset(data_dir, transformations=train_transform, split="train")
    #train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Validation dataset with defining split
    #val_dataset = CustomDataset(data_dir, transformations=val_transform, split="val")
    #val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # for images, masks in train_dataloader:
    #     # Use the train data here for training
    #     print(f"Image: {images.shape}")
    #     print(f"Mask: {masks.shape}")

    # Train dataset with pre-split
    split_train = SatelliteDataset(data_dir="./data/Train", pre_split=True)
    split_train_loader = DataLoader(split_train, batch_size=4, shuffle=False)

    # Test dataset with pre-split
    split_test = SatelliteDataset(data_dir="./data/Val", pre_split=True)
    split_test_loader = DataLoader(split_train, batch_size=4, shuffle=False)

    print(split_train.__len__())
    print(split_test.__len__()) """