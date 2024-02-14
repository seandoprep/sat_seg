import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import spectral
import spectral.io.envi as envi 
import torch.nn.functional as F

from glob import glob
from typing import Any

SMOOTH = 1e-8

def calculate_metrics(pred_mask: Any, true_mask: Any) -> torch.Tensor:
    '''
    Calculate Metrics
    '''
    pred_mask = pred_mask.view(-1).float()
    true_mask = true_mask.view(-1).float()
    eps=1e-5

    # Overlap Metrics
    tp = torch.sum(pred_mask * true_mask)  # TP
    fp = torch.sum(pred_mask * (1 - true_mask))  # FP
    fn = torch.sum((1 - pred_mask) * true_mask)  # FN
    tn = torch.sum((1 - pred_mask) * (1 - true_mask))  # TN   

    iou = (tp + eps) / (tp + fp + fn + eps) 
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = ((precision * recall + eps)/(precision + recall + eps))

    return iou.item(), dice.item(), pixel_acc.item(), f1.item()


""" def save_image_per_epochs( needs to be modified
        pred_mask: Any,
        true_mask: Any
        ) -> None:
    '''
    Save evaluation results 
    '''

    prediction = np.expand_dims(prediction, axis=-1)
    prediction = np.concatenate([prediction, prediction, prediction], axis=-1)

    overlay = np.multiply(image, prediction)
    prediction = prediction * 255

    overlay_img = np.concatenate([image, line, mask, line, prediction, line, overlay], axis=1)

    return """

def visualize_training_log(training_logs_csv: str, img_save_path: str):
    '''
    Show and Save training log visualization img
    '''
    training_log = pd.read_csv(training_logs_csv)
    epochs = training_log['Epoch']
    loss_train = training_log['Avg Train Loss']
    loss_val = training_log['Avg Val Loss']
    IoU_train = training_log['Avg IoU Train']
    IoU_val = training_log['Avg IoU Val']
    pixacc_train = training_log['Avg Pix Acc Train']
    pixacc_val = training_log['Avg Pix Acc Val']
    dice_train = training_log['Avg Dice Coeff Train']
    dice_val = training_log['Avg Dice Coeff Val']
    f1_train = training_log['Avg F1 Train']
    f1_val = training_log['Avg F1 Val']
    lr = training_log['Learning Rate']

    plt.figure(figsize=(28,16))

    # Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, loss_train)
    plt.plot(epochs, loss_val)
    plt.title('Train/Val Loss')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend(('val', 'train'))

    # IoU
    plt.subplot(2, 3, 2)
    plt.plot(epochs, IoU_train)
    plt.plot(epochs, IoU_val)
    plt.title('Train/Val IoU')
    plt.xlabel('epochs')
    plt.ylabel('IoU')
    plt.legend(('val', 'train'))

    # Pixel accuracy
    plt.subplot(2, 3, 3)
    plt.plot(epochs, pixacc_train)
    plt.plot(epochs, pixacc_val)
    plt.title('Train/Val pixacc')
    plt.xlabel('epochs')
    plt.ylabel('pixacc')
    plt.legend(('val', 'train'))

    # Dice score
    plt.subplot(2, 3, 4)
    plt.plot(epochs, dice_train)
    plt.plot(epochs, dice_val)
    plt.title('Train/Val dice')
    plt.xlabel('epochs')
    plt.ylabel('dice')
    plt.legend(('val', 'train'))

    # Learning rate
    plt.subplot(2, 3, 5)
    plt.plot(epochs, f1_train)
    plt.plot(epochs, f1_val)
    plt.title('Train/Val F1')
    plt.xlabel('epochs')
    plt.ylabel('f1')
    plt.legend(('val', 'train'))

    # Learning rate
    plt.subplot(2, 3, 6)
    plt.plot(epochs, lr)
    plt.title('Learning rate')
    plt.xlabel('epochs')
    plt.ylabel('learning_rate')

    plt.savefig(os.path.join(img_save_path, 'Training_log.png'))


def gpu_test():
    '''
    Check GPU availability
    '''
    if torch.version.cuda is None:
        print('Pytorch with CUDA is not ready')
    else :
        print('Pytorch Version : {}'.format(torch.version.cuda))

    if torch.cuda.is_available():
        print('CUDA is currently available')
    else: 
        print('CUDA is currently unavailable')


# Pad and Crop 
def pad_crop(original_array : np.ndarray, split_size : int):
    
    _, original_height, original_width = original_array.shape

    # Padding 
    X_num = (original_width - split_size) // split_size + 1
    Y_num = (original_height - split_size) // split_size + 1

    pad_x = (split_size * (X_num) + split_size) - original_width
    pad_y = (split_size * (Y_num) + split_size) - original_height

    padded_array = np.pad(original_array, ((0,0),(0,pad_y),(0,pad_x)), 'constant', constant_values=0)
    _, padded_height, padded_width = padded_array.shape

    stride_height = padded_height // split_size
    stride_width = padded_width // split_size

    # Cropping
    cropped_images = []
    for i in range(stride_height):
        for j in range(stride_width):
            start_x = i * split_size
            start_y = j * split_size
            end_x = start_x + split_size
            end_y = start_y + split_size

            cropped_image = padded_array[:, start_x:end_x, start_y:end_y]
            cropped_images.append(cropped_image)

    return np.array(cropped_images)


def read_envi_file(img_path):
    hdr_files = sorted(glob(os.path.join(img_path, "*.hdr")))
    img_files = sorted(glob(os.path.join(img_path, "*.img")))
    band_nums = len(hdr_files)

    envi_data = []
    for i in range(band_nums):
        envi_hdr_path = hdr_files[i]
        envi_img_path = img_files[i]

        data = envi.open(envi_hdr_path, envi_img_path)
        img = np.array(data.load())[:,:,0]
        envi_data.append(img)

    return np.array(envi_data)
