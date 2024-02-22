import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import spectral
import spectral.io.envi as envi 
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

from glob import glob
from typing import Any

SMOOTH = 1e-8

def calculate_metrics(pred_mask: Any, true_mask: Any) -> torch.Tensor:
    '''
    Calculate Metrics

    Metrics : IOU, Dice score, Pixel Accuracy, Precision, Recall, F1 score.
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


def visualize_train( 
        original_img: Any,
        pred_mask: Any,
        true_mask: Any,
        img_save_path : str,
        epoch : str,
        iter : str,
        ) -> None:
    '''
    Visualize training process per epoch and Save it.
    '''
    original_img_cpu = original_img[0].cpu().numpy()
    pred_mask_binary = F.sigmoid(pred_mask[0, 0]) > 0.5

    band_1 = original_img_cpu[0,:,:] 
    band_2 = original_img_cpu[1,:,:] 
    band_3 = original_img_cpu[2,:,:] 
    pred = pred_mask_binary.cpu().detach().numpy()
    pred = np.expand_dims(pred, axis=2)
    true = true_mask[0].cpu().detach().numpy()
    overlay = np.multiply(pred, true)

    #print("\nPrediction : \n", pred)

    plt.figure(figsize=(28,16))

    # 
    plt.subplot(2, 3, 1)
    plt.imshow(band_1, cmap='gray')
    plt.title('Band 1')

    # 
    plt.subplot(2, 3, 2)
    plt.imshow(band_2, cmap='gray')
    plt.title('Band 2')

    # Pixel accuracy
    plt.subplot(2, 3, 3)
    plt.imshow(band_3, cmap='gray')
    plt.title('Band 3')

    # Prediction
    plt.subplot(2, 3, 4)
    plt.imshow(pred, cmap='gray')
    plt.title('Prediction')

    # True Mask
    plt.subplot(2, 3, 5)
    plt.imshow(true, cmap='gray')
    plt.title('True Mask')

    # Overlay
    plt.subplot(2, 3, 6)
    plt.imshow(overlay, cmap='gray')
    plt.title('Overlay')

    plt.savefig(os.path.join(img_save_path, 'Training_result_epoch_{}_iter_{}.png'.format(epoch, iter)))
    plt.close()

def visualize_training_log(training_logs_csv: str, img_save_path: str):
    '''
    Visualize training log and Save it.
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
    plt.close()


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
    '''
    Pad and Crop Large Satellite Images for deep learning training.
    '''
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

# Band Normalization
def band_norm(band : np.array, norm_type : str, value_check : bool):
    '''
    Band Normalization for Satellite Image. 

    Tips : 
    1) Negative values are changed to Positive values for deep learning training
    2) norm_type should be one of linear_norm, or dynamic_world_norm
    3) Modify boundary values as necessary
    4) This code is suited for Input Image which is already Land/Sea Masked(Land value : 0)
     
    Need to add z-score, std normalization
    Reference : https://medium.com/sentinel-hub/how-to-normalize-satellite-images-for-deep-learning-d5b668c885af
    '''
    SMOOTH = 1e-5

    if np.any(band < 0):
        band_abs = band - np.min(band)
        #band_mean = np.mean(band_abs[band_abs != -np.min(band)])
        #band_abs[band_abs == -np.min(band)] = band_mean
    else:
        band_abs = band

    if norm_type == 'linear_norm':
        input_band_lower_bound, input_band_upper_bound = np.percentile(band_abs[band_abs != -np.min(band)], 1), np.percentile(band_abs[band_abs != -np.min(band)], 99)
        input_band_range = input_band_upper_bound - input_band_lower_bound

        band_norm = (band_abs - input_band_lower_bound) / input_band_range  # Percentile Normalization
        band_norm = np.clip((band_norm) / np.max(band_norm), 0, 1)  # Let Value Range : [0, 1]

    elif norm_type == 'dynamic_world_norm':
    
        def sigmoid(x):
            return 1 / (1 + np.exp(-(x+SMOOTH)))

        band_log = np.log1p(band_abs)
        band_log_for_percentile = np.log1p(band_abs[band_abs != -np.min(band)])
        #band_log_for_percentile = np.log1p(band_abs[band_abs != band_mean])

        input_band_lower_bound, input_band_upper_bound = np.percentile(band_log_for_percentile, 30), np.percentile(band_log_for_percentile, 70)  # Percentile Normalization
        input_band_range = input_band_upper_bound - input_band_lower_bound

        band_norm = sigmoid((band_log - input_band_lower_bound) / (input_band_range))  # Let Value Range : [0, 1] by Sigmoid Operation
    
    elif norm_type == 'mask_norm':
        band_norm = band_abs / 255.0

    else:
        raise Exception("norm_type should be one of 'linear_norm', or 'dynamic_world_norm'.")

    if value_check:
        print("Band Value :\n", band)
        print("Band Min Max :", np.min(band), np.max(band))
        print("Band abs Value :\n", band_abs)
        print("Band abs Min Max :", np.min(band_abs), np.max(band_abs))
        print("Input Lower Bound :", input_band_lower_bound)
        print("Input Upper Bound :", input_band_upper_bound)
        print("Band Norm Value :", band_norm)
        print("Band Norm Min Max :", np.min(band_norm), np.max(band_norm))
        print('--------------------------------------------------')

    return band_norm

# Read ENVI file Format
def read_envi_file(img_path, norm = True, norm_type = 'linear_norm'):
    '''
    Read ENVI file Format and return it as numpy array type.
    '''
    hdr_files = sorted(glob(os.path.join(img_path, "*.hdr")))
    img_files = sorted(glob(os.path.join(img_path, "*.img")))
    band_nums = len(hdr_files)

    envi_data = []
    for i in range(band_nums):
        envi_hdr_path = hdr_files[i]
        envi_img_path = img_files[i]

        data = envi.open(envi_hdr_path, envi_img_path)
        if norm:
            img = np.array(data.load())[:,:,0]
            img = band_norm(img, norm_type, False)
        else:
            img = np.array(data.load())[:,:,0]
            
        envi_data.append(img)

    return np.array(envi_data)

def find_arrays_with_object(arrays_list):
    indices_with_one = [index for index, array in enumerate(arrays_list) if np.any(array > 0)]

    return indices_with_one

# Randomness Control
def set_seed(random_seed : int):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    np.random.seed(random_seed)
    random.seed(random_seed)

    # We can fully control randomness, but speed will be slow
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
