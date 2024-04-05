import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import fiona
import netCDF4 as nc 
import cv2

from shapely.geometry import Polygon, mapping
from collections import defaultdict
from shapely import MultiPolygon
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

    return np.array(labeled_image)


def mask_to_polygons(mask, output_shapefile, lon_grid, lat_grid, epsilon=10., min_area=20.):
    """Convert a mask ndarray (binarized image) to Multipolygons"""
    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(contours[idx])

    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(contours):
        coords = []
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            for point in cnt:
                coord = [lon_grid[point[0][0]-1], lat_grid[point[0][1]-1]]
                coords.append(coord)
            poly = [coords]
            print(poly)
            all_polygons.append(poly)

    schema = {
        'geometry': 'MultiPolygon',
        'properties' : {}
    }

    # Create the Shapefile with the defined schema
    with fiona.open(output_shapefile, 'w', 'ESRI Shapefile', schema, crs = "EPSG:4326") as output:
        # Write to Shapefile
        output.write({
            'geometry': {'type' : 'MultiPolygon', 'coordinates' : all_polygons},
            'properties' : {}
        })

    return 