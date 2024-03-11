import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torchvision.transforms as transforms

from typing import Any
from torch.utils.data import Dataset
from utils.util import pad_crop, read_envi_file, find_arrays_with_object


class SatelliteDataset(Dataset):
    def __init__(
        self,
        data_dir: str = None,
        split: str = "train",
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        transform : transforms = None,
    ) -> None:
        if not os.path.exists(data_dir):
            raise ValueError(f'Provided data_dir: "{data_dir}" does not exist.')
        
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "Image")
        self.mask_dir = os.path.join(data_dir, "Mask")
        self.image_list = pad_crop(read_envi_file(self.image_dir, True, 'dynamic_world_norm'), 224)
        self.mask_list = pad_crop(read_envi_file(self.mask_dir, True, 'mask_norm'), 224)
        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.transform = transform

        #num_samples = len(self.image_list)
        #indices = list(range(num_samples))

        indices = find_arrays_with_object(self.mask_list)
        
        cnt = 0
        for num in range(len(self.image_list)):   
            cnt += 1 
            if num in indices:
                pass
            else:
                indices.append(num)    
            if cnt == 25:
                break

        num_samples = len(indices)

        # Data Split
        np.random.shuffle(indices)
        num_val_samples = int(self.val_ratio * num_samples)
        num_test_samples = int(self.test_ratio * num_samples)
        if self.split == "train":
            self.indices = indices[:-num_val_samples-num_test_samples]
        elif self.split == "val":
            self.indices = indices[-num_val_samples-num_test_samples:-num_test_samples]
        elif self.split == 'test':
            self.indices = indices[-num_test_samples:]
        else:
            raise ValueError("Invalid split value. Use 'train', 'val' or 'test'.")

 
    def __len__(self) -> int:
        return len(self.indices)


    def __getitem__(self, idx: Any) -> Any:
        img_idx = self.indices[idx]
        img = self.image_list[img_idx]
        mask = self.mask_list[img_idx]
        
        padded_img = np.pad(img, ((0,0),(16,16),(16,16)), 'constant', constant_values=0).swapaxes(0,2)
        padded_mask = np.pad(mask, ((0,0),(16,16),(16,16)), 'constant', constant_values=0).swapaxes(0,2)

        if self.transform:
            augmentations = self.transform(image=padded_img, mask=padded_mask)
            processed_img = augmentations["image"]
            processed_mask = augmentations["mask"]
        else:
            processed_img = transforms.ToTensor(padded_img)
            processed_mask = transforms.ToTensor(padded_mask)

        return processed_img, processed_mask

class InferenceDataset(Dataset):
    def __init__(
        self,
        data_dir: str = None,
        transform : transforms = None,
    ) -> None:
        if not os.path.exists(data_dir):
            raise ValueError(f'Provided data_dir: "{data_dir}" does not exist.')
        
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "Image")
        self.mask_dir = os.path.join(data_dir, "Mask")
        self.image_list = pad_crop(read_envi_file(self.image_dir, True, 'dynamic_world_norm'), 224)
        self.mask_list = pad_crop(read_envi_file(self.mask_dir, True, 'mask_norm'), 224)
        self.transform = transform
        self.indices = list(range(len(self.image_list)))
        #self.indices = find_arrays_with_object(self.mask_list)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: Any) -> Any:
        img_idx = self.indices[idx]
        img = self.image_list[img_idx]
        mask = self.mask_list[img_idx]
        
        padded_img = np.pad(img, ((0,0),(16,16),(16,16)), 'constant', constant_values=0).swapaxes(0,2).swapaxes(0,1)
        padded_mask = np.pad(mask, ((0,0),(16,16),(16,16)), 'constant', constant_values=0).swapaxes(0,2).swapaxes(0,1)

        if self.transform:
            augmentations = self.transform(image=padded_img, mask=padded_mask)
            processed_img = augmentations["image"]
            processed_mask = augmentations["mask"]
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