import numpy as np
import os
import matplotlib.pyplot as plt 
import click
import cv2

from utils import band_norm, pad_crop, read_envi_file, find_arrays_with_object

@click.command()
@click.option("-D", "--data-dir", type=str, default='data\\Train\\ENVI', help="Path for Data Directory")

def main(
    data_dir: str,
) -> None:
    """
    Visualize Script for 3 channel Satellite Image.
    """
    click.secho(message="🚀 Visualizing...", fg="green", nl=True)

    # Read Image, Mask
    data_dir = data_dir
    image_dir = os.path.join(data_dir, "Image")
    mask_dir = os.path.join(data_dir, "Mask")
    image_list = pad_crop(read_envi_file(image_dir, True), 224)
    mask_list = pad_crop(read_envi_file(mask_dir, False), 224)
    
    # Random Sampling
    indices = find_arrays_with_object(mask_list)

    np.random.seed(59)
    np.random.shuffle(indices)
    sample = indices[0]

    # Visualization
    plt.figure(figsize=(50,50))
    cols, rows = 3, 5
    img, mask = image_list[sample], mask_list[sample]
    image_np = np.array(img, dtype=np.float32)
    mask_np = np.array(mask, dtype=np.float32)

    for i in range(cols):
        org_band = image_np[i,:,:]
        linear_norm = band_norm(org_band, 'linear_norm')
        dynamic_world_norm = band_norm(org_band, 'dynamic_world_norm')
        true_mask = mask_np[0,:,:]

        results = [org_band, linear_norm, dynamic_world_norm, true_mask]
        labels = ['Band_{}'.format(i+1), 'Linear Normalization', 'Dynamic Normalization', 
                'True Mask', 'Band Histogram']
        
        for j in range(rows):
            plt.subplot(cols, rows, 5*i+j+1)
            if labels[j] == 'Band Histogram':

                linear_norm_1d = np.array(linear_norm).ravel()
                dynamic_world_norm_1d = np.array(dynamic_world_norm).ravel()

                plt.hist([linear_norm_1d, dynamic_world_norm_1d], 
                         bins = 150,
                         range = [0.0,1.0],
                         color = ['green', 'blue'],
                         label = [ 'linear_norm', 'dynamic_world_norm'],
                         histtype = 'step',
                         stacked=True)
                plt.xlim()
                plt.ylim()
                plt.legend()
            else:
                plt.imshow(results[j], cmap='gray')
            plt.axis('off')
            plt.title(labels[j])
            
    plt.show()
    click.secho(message="🚀 End Visualizing...", fg="red", nl=True)
    plt.close()

if __name__ == "__main__":
    main()