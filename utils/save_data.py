import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import fiona
import netCDF4 as nc 

from shapely.geometry import Polygon, mapping
from netCDF4 import Dataset


def save_nc(save_path : str, prediction_array : np.array, lat_grid : np.array, lon_grid : np.array) :
    '''
    Convert prediction mask numpy array to nc file format data.

    save_path : Path where nc file will be saved
    original_array_path : Original nc file for geocoordinate information
    prediction_array : Numpy array that needs to be converted
    '''
    save_path = os.path.join(save_path, 'Inference_output.nc')
    height, width = prediction_array.shape

    # Set NC file
    ncfile = Dataset(save_path, mode='w', format='NETCDF4') 
    lat_dim = ncfile.createDimension('lat', height) # latitude axis
    lon_dim = ncfile.createDimension('lon', width) # longitude axis

    # Title of NC file
    ncfile.title='inference_result'

    # Latitude 
    lat = ncfile.createVariable('lat', np.float32, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'

    # Longitude 
    lon = ncfile.createVariable('lon', np.float32, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'

    # Define a 3D variable to hold the data
    inf_result = ncfile.createVariable('prediction',np.int64,('lat','lon')) # Note: unlimited dimension is leftmost

    # Add Coordinate Information
    lat[:] = lat_grid 
    lon[:] = lon_grid
    inf_result[:,:] = prediction_array
    
    ncfile.close()


def label_binary_image(binary_array : np.array):
    height, width = binary_array.shape
    labeled_image = [[0 for _ in range(width)] for _ in range(height)]
    label = 1

    # 8-directional movement offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1), (1, 0), (1, 1)]

    # Function to check if a pixel is within the image bounds
    def is_valid_pixel(x, y):
        return 0 <= x < width and 0 <= y < height

    # Function to check if a pixel has already been labeled
    def is_labeled(x, y):
        return labeled_image[y][x] != 0

    # Function to get all unlabeled neighboring pixels
    def get_unlabeled_neighbors(x, y):
        neighbors = []
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if is_valid_pixel(nx, ny) and not is_labeled(nx, ny) and binary_array[ny][nx] == 1:
                neighbors.append((nx, ny))
        return neighbors

    # Function to perform label propagation from a seed pixel
    def propagate_label(seed_x, seed_y):
        stack = [(seed_x, seed_y)]
        while stack:
            x, y = stack.pop()
            labeled_image[y][x] = label
            neighbors = get_unlabeled_neighbors(x, y)
            stack.extend(neighbors)

    # Main loop to label connected components in the binary image
    for y in range(height):
        for x in range(width):
            if binary_array[y][x] == 1 and not is_labeled(x, y):
                propagate_label(x, y)
                label += 1

    return labeled_image


def create_polygon_shapefile(labeled_image, output_shapefile, lat_grid, lon_grid):
    # Define the schema for the Shapefile
    schema = {
        'geometry': 'Polygon',
        'properties': {'label': 'int'}
    }

    # Create the Shapefile with the defined schema
    with fiona.open(output_shapefile, 'w', 'ESRI Shapefile', schema) as output:
        height, width = len(lat_grid), len(lon_grid)

        # Iterate through the labeled image and create polygons
        for lat in range(lat_grid):
            for lon in range(lon_grid):
                label = labeled_image[lat][lon]
                if label != 0:  # Skip unlabeled pixels
                    # Define the coordinates of the polygon
                    # Assuming each pixel represents a square of unit size
                    x_min = lon
                    y_min = height - lat - 1  # Flip y-coordinate to match image coordinate system
                    x_max = lon + 1
                    y_max = y_min + 1

                    # Create a Shapely Polygon object
                    polygon = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])

                    # Add the polygon to the Shapefile with its label as a property
                    output.write({
                        'geometry': mapping(polygon),
                        'properties': {'label': label}
                    })