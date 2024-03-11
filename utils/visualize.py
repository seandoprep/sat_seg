import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

from typing import Any


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

    plt.savefig(os.path.join(img_save_path, 'Training_result_epoch_{}_iter_{}.png'.format(epoch, iter)))
    plt.close()


def visualize_test( 
        original_img: Any,
        pred_mask: Any,
        true_mask: Any,
        img_save_path : str,
        num : int
        ) -> None:
    '''
    Visualize test process per image and Save it.
    '''
    original_img_cpu = original_img[0].cpu().numpy()
    pred_mask_binary = F.sigmoid(pred_mask[0, 0]) > 0.5

    band_1 = original_img_cpu[0,:,:] 
    band_2 = original_img_cpu[1,:,:] 
    band_3 = original_img_cpu[2,:,:] 
    pred = pred_mask_binary.cpu().detach().numpy()
    pred = np.expand_dims(pred, axis=2)
    true = true_mask[0].cpu().detach().numpy()

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

    plt.savefig(os.path.join(img_save_path, 'Test_result_{}.png'.format(num)))
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