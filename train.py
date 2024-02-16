import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import csv
import sys
import click
import traceback
import torch
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import SatelliteDataset
from models.deeplabv3plus import DeepLabV3Plus
from models.unet import UNet
from models.resunetplusplus import ResUnetPlusPlus
from models.transunet import TransUNet
from loss import DiceLoss, DiceBCELoss, IoULoss, FocalLoss, TverskyLoss
from utils import visualize_training_log, calculate_metrics, gpu_test, visualize_train
from datetime import datetime
from scheduler import CosineAnnealingWarmUpRestarts

INPUT_CHANNEL_NUM = 3
INPUT = (256, 256)
CLASSES = 1  # For Binary Segmentatoin


@click.command()
@click.option("-D", "--data-dir", type=str, default='data\\Train\\ENVI', help="Path for Data Directory")
@click.option(
    "-M",
    "--model-name",
    type=str,
    default='deeplabv3plus',
    help="Choose models for Binary Segmentation. unet, deeplabv3plus, resunetplusplus, and transunet are now available",
)
@click.option(
    "-E",
    "--num-epochs",
    type=int,
    default=100,
    help="Number of epochs to train the model for. Default - 100",
)
@click.option(
    "-L",
    "--learning-rate",
    type=float,
    default=0,
    help="Learning Rate for model. Default - 1e-3",
)
@click.option(
    "-B",
    "--batch-size",
    type=int,
    default=16,
    help="Batch size of data for training. Default - 32",
)
@click.option(
    "-S",
    "--early-stop",
    type=bool,
    default=True,
    help="Stop training if val_loss hasn't improved for a certain no. of epochs. Default - True",
)
def main(
    data_dir: str,
    model_name : str,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    early_stop: bool,
) -> None:
    """
    Training Script for DeepLabV3+ with ResNet50 Encoder for Binary Segmentation.
    Please make sure your data is structured according to the folder structure specified in the Github Repository.
    See: https://github.com/mukund-ks/DeepLabV3Plus-PyTorch

    Refer to the Options below for usage.
    """
    click.secho(message="🚀 Training...", fg="blue", nl=True)


    try:
        train_dataset = SatelliteDataset(
            data_dir=data_dir, split="train", transform=True
        )
        val_dataset = SatelliteDataset(
            data_dir=data_dir, split="val", transform=True
        )
    except Exception as _:
        click.secho(message="\n❗ Error \n", fg="red")
        click.secho(message=traceback.format_exc(), fg="yellow")
        sys.exit("Non-Existent Data Dir")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Defining Model(Only channel 1 or 3 img data can be used)
    if model_name == 'unet':
        model = UNet(in_channels=3, num_classes=CLASSES)  # Can handle 1, 3 channel img data
    elif model_name == 'deeplabv3plus':
        model = DeepLabV3Plus(num_classes=CLASSES)  # Only handle 3 channel img data because of pretrained backbone
    elif model_name == 'resunetplusplus':
        model = ResUnetPlusPlus(in_channels=3, num_classes = CLASSES)  # Can handle 1, 3 channel img data
    elif model_name == 'transunet':
        model = TransUNet(256, 3, 128, 4, 512, 8, 16, CLASSES)  # Can handle 1, 3 channel img data
    
    # Check GPU Availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_test()
    model.to(device)

    # Loss Function Setting
    criterion = DiceLoss()
    #criterion = nn.BCELoss()
    #criterion = DiceBCELoss()
    #criterion = IoULoss()
    #criterion = FocalLoss()
    #criterion = TverskyLoss()

    # Optimizer & Scheduler    
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer, mode="min", patience=3, factor=0.1, verbose=True
    #)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)    

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=150, T_mult=1, eta_max=0.1,  T_up=10, gamma=0.5)

    # For Early-Stopping
    patience_epochs = 10
    no_improvement_epochs = 0

    # Logging
    csv_file_path = "weights/train_result/training_logs.csv"
    csv_header = [
        "Epoch",
        "Avg Train Loss",
        "Avg Val Loss",
        "Avg IoU Train",
        "Avg IoU Val",
        "Avg Pix Acc Train",
        "Avg Pix Acc Val",
        "Avg Dice Coeff Train",
        "Avg Dice Coeff Val",
        "Avg F1 Train",
        "Avg F1 Val",
        "Learning Rate",
    ]

    # For saving best model
    best_val_loss = float("inf")

    click.echo(
        f"\n{click.style(text=f'Train Size: ', fg='blue')}{train_dataset.__len__()}\t{click.style(text=f'val Size: ', fg='blue')}{val_dataset.__len__()}\n"
    )

    # Main loop
    with open(csv_file_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(csv_header)

        for epoch in range(num_epochs):
            # TRAINING
            model.train()
            train_loss = 0.0
            total_iou_train = 0.0
            total_pixel_accuracy_train = 0.0
            total_dice_coefficient_train = 0.0
            total_f1_train = 0.0

            train_dataloader = tqdm(
                train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
            )

            current_lr = optimizer.param_groups[0]["lr"]

            iter = 0
            for images, masks in train_dataloader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                if epoch % 5 == 0:
                    if iter % 50 == 0:
                        visualize_train(images, outputs, masks, 
                                        img_save_path= 'outputs/train_output', 
                                        epoch = str(epoch), iter = str(iter))
                        click.echo(
                            f"\n{click.style(text=f'Saved Train Process', fg='green')}\t{click.style(text=f'Epoch : ', fg='green')}{str(epoch)}\t{click.style(text=f'Iter : ', fg='green')}{str(iter)}"
                            )

                t_loss = criterion(outputs, masks)

                t_loss.backward()
                optimizer.step()

                train_loss += t_loss.item()
                iter += 1

                # Calculating metrics for training
                with torch.no_grad():
                    pred_masks = outputs > 0.5
                    iou_train, dice_coefficient_train, pixel_accuracy_train, f1_train = calculate_metrics(
                        pred_masks, masks
                    )

                    total_iou_train += iou_train
                    total_dice_coefficient_train += dice_coefficient_train
                    total_pixel_accuracy_train += pixel_accuracy_train
                    total_f1_train += f1_train

                # Displaying metrics in the progress bar description
                train_dataloader.set_postfix(
                    loss=t_loss.item(),
                    train_iou=iou_train,
                    train_pix_acc=pixel_accuracy_train,
                    train_dice_coef=dice_coefficient_train,
                    train_f1=f1_train,
                    lr=current_lr,
                )

            train_loss /= len(train_dataloader)
            avg_iou_train = total_iou_train / len(train_dataloader)
            avg_pixel_accuracy_train = total_pixel_accuracy_train / len(train_dataloader)
            avg_dice_coefficient_train = total_dice_coefficient_train / len(train_dataloader)
            avg_f1_train = total_f1_train / len(train_dataloader)

            # VALIDATION
            model.eval()
            val_loss = 0.0
            total_iou_val = 0.0
            total_pixel_accuracy_val = 0.0
            total_dice_coefficient_val = 0.0
            total_f1_val = 0.0

            val_dataloader = tqdm(val_dataloader, desc=f"Validation", unit="batch")

            with torch.no_grad():
                for images, masks in val_dataloader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)

                    v_loss = criterion(outputs, masks)
                    val_loss += v_loss.item()

                    # Calculating metrics for Validation
                    pred_masks = outputs > 0.5
                    iou_val, dice_coefficient_val, pixel_accuracy_val, f1_val = calculate_metrics(
                        pred_masks, masks
                    )

                    total_iou_val += iou_val
                    total_pixel_accuracy_val += pixel_accuracy_val
                    total_dice_coefficient_val += dice_coefficient_val
                    total_f1_val += f1_val

                    # Displaying metrics in progress bar description
                    val_dataloader.set_postfix(
                        val_loss=v_loss.item(),
                        val_iou=iou_val,
                        val_pix_acc=pixel_accuracy_val,
                        val_dice_coef=dice_coefficient_val,
                        val_f1=f1_val,
                        lr=current_lr,
                    )

            val_loss /= len(val_dataloader)
            avg_iou_val = total_iou_val / len(val_dataloader)
            avg_pixel_accuracy_val = total_pixel_accuracy_val / len(val_dataloader)
            avg_dice_coefficient_val = total_dice_coefficient_val / len(val_dataloader)
            avg_f1_val = total_f1_val / len(val_dataloader)

            scheduler.step(val_loss)

            print(
                f"\nEpoch {epoch + 1}/{num_epochs}\n"
                f"Avg Train Loss: {train_loss:.4f}\n"
                f"Avg Validation Loss: {val_loss:.4f}\n"
                f"Avg IoU Train: {avg_iou_train:.4f}\n"
                f"Avg IoU Val: {avg_iou_val:.4f}\n"
                f"Avg Pix Acc Train: {avg_dice_coefficient_train:.4f}\n"
                f"Avg Pix Acc Val: {avg_pixel_accuracy_val:.4f}\n"
                f"Avg Dice Coeff Train: {avg_dice_coefficient_train:.4f}\n"
                f"Avg Dice Coeff Val: {avg_dice_coefficient_val:.4f}\n"
                f"Avg F1 Train: {avg_f1_train:.4f}\n"
                f"Avg F1 Val: {avg_f1_val:.4f}\n"
                f"Current LR: {current_lr}\n"
                f"{'-'*50}"
            )

            # Saving best model
            if val_loss < best_val_loss:
                no_improvement_epochs = 0
                click.secho(
                    message=f"\n👀 val_loss improved from {best_val_loss:.4f} to {val_loss:.4f}\n",
                    fg="green",
                )
                best_val_loss = val_loss
                torch.save(model.state_dict(), "weights/train_result/best_model.pth") 
                click.secho(message="Saved Best Model! 🙌\n", fg="green")
                print(f"{'-'*50}")
            else:
                no_improvement_epochs += 1
                click.secho(
                    message=f"\nval_loss did not improve from {best_val_loss:.4f}\n", fg="yellow"
                )
                print(f"{'-'*50}")

            # Append the training and validation logs to the CSV file
            csv_writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    val_loss,
                    avg_iou_train,
                    avg_iou_val,
                    avg_pixel_accuracy_train,
                    avg_pixel_accuracy_val,
                    avg_dice_coefficient_train,
                    avg_dice_coefficient_val,
                    avg_f1_train,
                    avg_f1_val,
                    current_lr,
                ]
            )

            # Early-Stopping
            if early_stop:
                if no_improvement_epochs >= patience_epochs:
                    click.secho(
                        message=f"\nEarly Stopping: val_loss did not improve for {patience_epochs} epochs.\n",
                        fg="red",
                    )
                    break

    click.secho(message="🎉 Training Done!", fg="blue", nl=True)

    # Save train result
    train_base_dir = 'outputs/train_output'
    now = datetime.now()
    folder_name = now.strftime("%Y_%m_%d_%H_%M_%S") + model_name
    train_output_dir = os.path.join(train_base_dir, folder_name)

    try:
        os.makedirs(train_output_dir, exist_ok=True)
        click.secho(message="Train output folder was successfully created\n", fg="blue")
    except OSError as e:
        click.secho(message="\n❗ Error\n", fg="red")
        sys.exit("OSError while creating output data dir")

    visualize_training_log(csv_file_path, train_output_dir)

    return


if __name__ == "__main__":
    main()
