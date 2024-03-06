import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import torch
import click
import traceback
import albumentations as A
import numpy as np
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from models.deeplabv3plus import DeepLabV3Plus
from models.unet import UNet
from models.resunetplusplus import ResUnetPlusPlus
from models.transunet import TransUNet
from dataset import InferenceDataset
from utils import set_seed, gpu_test, unpad, read_envi_file, restore_img, pad_crop
from datetime import datetime
from PIL import Image

import torch.nn.functional as F
import matplotlib.pyplot as plt


INPUT_CHANNEL_NUM = 3
INPUT = (256, 256)
CLASSES = 1  # For Binary Segmentatoin


@click.command()
@click.option("-D", "--data-dir", type=str, default='data\\Train\\ENVI', required=True, help="Path for Data Directory")
@click.option(
    "-M",
    "--model-name",
    type=str,
    default='deeplabv3plus',
    help="Choose models for Binary Segmentation. unet, deeplabv3plus, resunetplusplus, and transunet are now available.",
)
@click.option(
    "-S",
    "--model-path",
    type=str,
    default='./weights/train_result/best_model.pth',
    help="Path for pretrained model weight file",
)
@click.option(
    "-B",
    "--batch-size",
    type=int,
    default=1,
    help="Batch size of data for Inference. Default - 8",
)
def main(
    data_dir: str,
    model_name : str,
    model_path : str,
    batch_size : int) -> None:
    """
    Inference Script for DeepLabV3+ with ResNet50 Encoder for Binary Segmentation.\n
    Please make sure your evaluation data is structured according to the folder structure specified in the Github Repository.\n
    See: https://github.com/mukund-ks/DeepLabV3Plus-PyTorch

    Refer to the Option(s) below for usage.
    """
    click.secho(message="🔎 Inference...", fg="blue")

    set_seed(99)
    custom_transform = A.Compose([
        ToTensorV2(),
    ])

    try:
        inference_dataset = InferenceDataset(data_dir=data_dir, transform=custom_transform)
        inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)
        click.echo(message=f"\n{click.style('Inference Size: ', fg='blue')}{inference_dataset.__len__()}\n")
        inference_dataloader = tqdm(inference_dataloader, desc="Inference", unit="image")
    except Exception as _:
        click.secho(message="\n❗ Error\n", fg="red")
        click.secho(message=traceback.format_exc(), fg="yellow")
        sys.exit("Non-Existent Data Dir")


    # Defining Model(Only channel 1 or 3 img data can be used)
    if model_name == 'unet':
        model = UNet(in_channels=3, num_classes=CLASSES)  # Can handle 1, 3 channel img data
    elif model_name == 'deeplabv3plus':
        model = DeepLabV3Plus(num_classes=CLASSES)  # Only handle 3 channel img data because of pretrained backbone
    elif model_name == 'resunetplusplus':
        model = ResUnetPlusPlus(in_channels=3, num_classes = CLASSES)  # Can handle 1, 3 channel img data
    elif model_name == 'transunet':
        model = TransUNet(256, 3, 128, 4, 512, 8, 16, CLASSES)  # Can handle 1, 3 channel img data

    # Load Trained Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_test()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Save test result
    inference_base_dir = 'outputs/inference_output'
    now = datetime.now()
    folder_name = now.strftime("%Y_%m_%d_%H_%M_%S") + model_name
    inference_output_dir = os.path.join(inference_base_dir, folder_name)

    try:
        os.makedirs(inference_output_dir, exist_ok=True)
        click.secho(message="Inference output folder was successfully created\n", fg="blue")
    except OSError as e:
        click.secho(message="\n❗ Error\n", fg="red")
        sys.exit("OSError while creating output data dir")

    # Main loop
    image_list = []
    pad_length = 16

    img_path = os.path.join(data_dir, 'Image')
    _, image_height, image_width = read_envi_file(img_path, True, 'dynamic_world_norm').shape

    with torch.no_grad():
        for i, (images, _) in enumerate(inference_dataloader):
            images = images.to(device)

            outputs = model(images)
            
            #for img_num in range(batch_size):
            #pred_mask_binary = F.sigmoid(outputs[img_num].squeeze()) > 0.5
            pred_mask_binary = F.sigmoid(outputs[0].squeeze()) > 0.5
            pred_mask_np = pred_mask_binary.cpu().detach().numpy()
            pred_mask_np = unpad(pred_mask_np, pad_length)
            image_list.append(pred_mask_np)

    # Restore Images
    restored_img = restore_img(image_list, image_height, image_width, 224)
    restored_img = np.array(restored_img, np.uint8) * 255
    img = Image.fromarray(restored_img)
    img.save((os.path.join(inference_output_dir, 'Inference_output.jpg')), 'JPEG')

    click.secho(message="🎉 Inference Done!", fg="blue")
    return

if __name__ == "__main__":
    main()