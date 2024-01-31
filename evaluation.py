import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import torch
import click
import traceback
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from models.deeplabv3plus import DeepLabV3Plus
from models.unet import UNet
from models.resunetplusplus import ResUnetPlusPlus
from models.transunet import TransUNet
from dataset import EvalDataset
from utils import save_image
from datetime import datetime

INPUT_CHANNEL_NUM = 1 
INPUT = (256, 256)
CLASSES = 1  # For Binary Segmentatoin


@click.command()
@click.option("-D", "--data-dir", type=str, required=True, help="Path for Data Directory")
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
def main(
    data_dir: str,
    model_name : str,
    model_path : str) -> None:
    """
    Evaluation Script for DeepLabV3+ with ResNet50 Encoder for Binary Segmentation.\n
    Please make sure your evaluation data is structured according to the folder structure specified in the Github Repository.\n
    See: https://github.com/mukund-ks/DeepLabV3Plus-PyTorch

    Refer to the Option(s) below for usage.
    """
    click.secho(message="🔎 Evaluation...", fg="blue")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Defining Model(Only channel 1 or 3 img data can be used)
    if model_name == 'unet':
        model = UNet(in_channels=3, num_classes=CLASSES)  # Can handle 1, 3 channel img data
    elif model_name == 'deeplabv3plus':
        model = DeepLabV3Plus(num_classes=CLASSES)  # Only handle 3 channel img data because of pretrained backbone
    elif model_name == 'resunetplusplus':
        model = ResUnetPlusPlus(in_channels=3, num_classes = CLASSES)  # Can handle 1, 3 channel img data
    elif model_name == 'transunet':
        model = TransUNet(256, 3, 128, 4, 512, 8, 16, CLASSES)  # Can handle 1, 3 channel img data
    

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Create output folder
    output_base_dir = "data/Test/Output"
    now = datetime.now()
    folder_name = now.strftime("%Y_%m_%d_%H_%M_%S") + model_name
    output_dir = os.path.join(output_base_dir, folder_name)
    img_output_dir = os.path.join(output_dir, 'Image')
    mask_output_dir = os.path.join(output_dir, 'Mask')
    overlay_output_dir = os.path.join(output_dir, "Overlayed")

    try:
        os.makedirs(img_output_dir, exist_ok=True)
        os.makedirs(mask_output_dir, exist_ok=True)
        os.makedirs(overlay_output_dir, exist_ok=True)
        click.secho(message="Evaluation output folders were successfully created\n", fg="blue")
    except OSError as e:
        click.secho(message="\n❗ Error\n", fg="red")
        sys.exit("OSError while creating output data dir")

    eval_transform = A.Compose(
        [
            A.Resize(INPUT[0], INPUT[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    try:
        eval_dataset = EvalDataset(data_dir=data_dir, transformations=eval_transform, INPUT_CHANNEL_NUM = 3)
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
        click.echo(message=f"\n{click.style('Evaluation Size: ', fg='blue')}{eval_dataset.__len__()}\n")
        eval_dataloader = tqdm(eval_dataloader, desc="Evaluation", unit="image")
    except Exception as _:
        click.secho(message="\n❗ Error\n", fg="red")
        click.secho(message=traceback.format_exc(), fg="yellow")
        sys.exit("Non-Existent Data Dir")

    with torch.no_grad():
        for i, (image, _) in enumerate(eval_dataloader):
            image = image.to(device)

            output = model(image)
            pred_mask = output < 0.45
            prediction = pred_mask.cpu().numpy()[0, 0]

            img_path = eval_dataset.image_filenames[i]
            mask_path = eval_dataset.mask_filenames[i]

            output_img_path = os.path.join(img_output_dir, f"img_output_{i}.png")
            output_mask_path = os.path.join(mask_output_dir, f"mask_output_{i}.png")
            output_overlay_path = os.path.join(overlay_output_dir, f"overlay_output_{i}.png")
            save_image(img_path, mask_path, prediction, output_img_path, output_mask_path, output_overlay_path)

    click.secho(message="🎉 Evaluation Done!", fg="blue")
    return


if __name__ == "__main__":
    main()
