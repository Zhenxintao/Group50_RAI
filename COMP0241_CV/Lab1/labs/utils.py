import numpy as np
import copy
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import random
from PIL import Image
from typing import Union
from sklearn.feature_extraction.image import check_array, _extract_patches, \
_compute_n_patches, check_random_state


def resize(img, scale: Union[float, int]) -> np.ndarray:
    """ Resize an image maintaining its proportions
    Args:
        fp (str): Path argument to image file
        scale (Union[float, int]): Percent as whole number of original image. eg. 53
    Returns:
        image (np.ndarray): Scaled image
    """
    _scale = lambda dim, s: int(dim * s / 100)
    height, width, channels = img.shape
    new_width: int = _scale(width, scale)
    new_height: int = _scale(height, scale)
    new_dim: tuple = (new_width, new_height)
    return cv2.resize(src=img, dsize=new_dim, interpolation=cv2.INTER_LINEAR)

def colourize(img):
    height, width = img.shape

    colors = []
    colors.append([])
    colors.append([])
    color = 1
    # Displaying distinct components with distinct colors
    coloured_img = Image.new("RGB", (width, height))
    coloured_data = coloured_img.load()

    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] > 0:
                if img[i][j] not in colors[0]:
                    colors[0].append(img[i][j])
                    colors[1].append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

                ind = colors[0].index(img[i][j])
                coloured_data[j, i] = colors[1][ind]

    return coloured_img


def binarize(img_array, threshold=130):
    for i in range(len(img_array)):
        for j in range(len(img_array[0])):
            if img_array[i][j] > threshold:
                img_array[i][j] = 0
            else:
                img_array[i][j] = 1
    return img_array


def draw_corners(image, corners_map):
    """Draw a point for each possible corner."""
    
    color_img = cv2.cvtColor(image[:,:,0], cv2.COLOR_GRAY2BGR)
    for each_corner in corners_map:
        cv2.circle(color_img, (each_corner[1], each_corner[0]), 1, (255,0,0), -1)
    return color_img

def load_image(path_name):
    image = cv2.imread(path_name)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

def apply_kernel(image, kernel):
    kernel_size = kernel.shape[0]

    padding_amount = int((kernel_size - 1) / 2)
    rows = image.shape[0] + 2 * padding_amount
    cols = image.shape[1] + 2 * padding_amount
    channels = image.shape[2]
    padded_image_placeholder = np.zeros((rows, cols, channels))
    padded_image_placeholder[padding_amount:rows-padding_amount, padding_amount:cols-padding_amount, :] = image

    filtered_image = np.zeros(image.shape)

    for each_channel in range(channels):
        padded_2d_image = padded_image_placeholder[:,:,each_channel]
        filtered_2d_image = filtered_image[:,:,each_channel]
        width = padded_2d_image.shape[0]
        height = padded_2d_image.shape[1]
        for i in range(width-kernel_size+1):
            for j in range(height-kernel_size+1):
                current_block = padded_2d_image[i:i+kernel_size, j:j+kernel_size]
                convoluted_value = np.sum(current_block * kernel)
                filtered_2d_image[i][j] = convoluted_value
        filtered_image[:,:,each_channel] = filtered_2d_image

    return filtered_image

def get_gaussian_filter(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    denom = 2 * np.pi * sigma * sigma
    samples = np.arange(-int(kernel_size/2), int(kernel_size/2) + 1)

    for i in range(len(samples)):
        for j in range(len(samples)):
            x = samples[i]
            y = samples[j]
            num = np.exp(-1*(((x*x) + (y*y)) / (2*sigma*sigma)))
            val = num / np.sqrt(denom)
            kernel[i][kernel_size - j - 1] = val
    kernel = kernel / kernel.sum()
    
    return kernel
