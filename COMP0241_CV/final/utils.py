import os
import re
import json

import cv2
import numpy as np


def load_params(path):
    """
    Load the parameters for all datasets from a json file.

    Args:
        path (str)

    Returns:
        Dict: params for all datasets
    """
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def get_images_in_folder(path):
    """
    Get all images in a given folder.

    Args:
        path: (str): Path to the image folder.

    Returns:
        List[Dict{
            "path": str,
            "frame": int,
            "fps": float,
            "timestamp": int,
        }]: A list of image dicts, including the image and video information.
    """
    imageList = []
    imageNames = os.listdir(path)
    if imageNames is []:
        raise ValueError("Image not found at the provided path.")

    for frame, imageName in enumerate(imageNames):
        imageInfo = {
            "path": os.path.join(path, imageName),
            "frame": frame,
        }
        if "frame" in imageName:
            imageInfo["fps"] = 30  # 31.96
        else:
            imageInfo["timestamp"] = int(imageName[:-4])

        imageList.append(imageInfo)
    return imageList


def get_datasets(paths, max_len=-1):
    """
    Generate the dataset.

    Args:
        paths (List[str]): folders of images

    Returns:
        List[Dict{
            "path": str,
            "images": List[Dict{
                "path": str,
                "frame": int,
                "fps": float,
                "timestamp": int,
            }]
        }]: A list of folder dicts, each has keys "path" and "images",
            represent the folder path and a list of image dicts.
    """
    datasets = []
    for path in paths:
        imageList = get_images_in_folder(path)

        if 0 < max_len < len(imageList):
            imageList = imageList[:max_len]

        datasets.append({
            "path": path,
            "images": imageList
        })
    return datasets


def dataset_to_image_pair(datasets, circles):
    """

    Args:
        datasets: List[Dict{
                "path": str,
                "images": List[Dict{
                    "path": str,
                    "frame": int,
                    "fps": float,
                    "timestamp": int,
                }]
            }]

    Returns:
        List[Dict{
            "path": str,
            "left": List[Dict{
                    "path": str,
                    "frame": int,
                    "fps": float,
                    "timestamp": int,
            ]}
            "right": List[Dict{
                "path": str,
                "frame": int,
                "fps": float,
                "timestamp": int,
            ]}
            "left_circle": Dict{
                "centers": ndarray,
                "masks": ndarray,
            }
            "right_circle": Dict{
                "centers": ndarray,
                "masks": ndarray,
            }
        }]
    """
    commonPrefix = {}
    prefix = {}
    for dataset in datasets:
        pre = re.split(r'\b(left|right)\b', dataset["path"])[0]
        if pre in prefix:
            commonPrefix[pre] = dataset
        else:
            prefix[pre] = dataset

    pairDatasets = []
    for pre in commonPrefix:
        dataset1 = commonPrefix[pre]
        dataset2 = prefix[pre]

        leftPath = os.path.join(pre, "left")
        rightPath = os.path.join(pre, "right")

        if "left" in dataset1["path"]:
            leftImage = dataset1["images"]
            rightImage = dataset2["images"]
        else:
            leftImage = dataset2["images"]
            rightImage = dataset1["images"]

        pairDataset = {
            "path": pre,
            "left": leftImage,
            "right": rightImage,
            "left_circle": circles[leftPath],
            "right_circle": circles[rightPath],
        }
        pairDatasets.append(pairDataset)
    return pairDatasets


class ImageReader:
    """
    A image reader including the image calibration function.

    Attributes:
        intrinsic (ndarray): Intrinsic Parameters of the camera, [3, 3]
        distortion (ndarray): Distortion Coefficients of the camera, [1, 5]
        mapper Dict{
                W * H: {
                    "mapx": mapx,
                    "mapy": mapy,
                }
            }: Precompute the undistortion maps (mapx, mapy) for each (W, H) to
               accelerate the processing of a large number of images.
        W, H (int): Default width and height of the image.
    """
    def __init__(self, path):
        self.intrinsic = None
        self.distortion = None
        self.mapper = {}

        self.W = 1280
        self.H = 800

        self.get_calibration(path)

    @property
    def fx(self):
        return self.intrinsic[0][0]

    @property
    def fy(self):
        return self.intrinsic[1][1]

    def get_calibration(self, path):
        """
        Load the camera calibration parameters.

        Args:
            path (str): Calibration file path, *.npz
        """
        data = np.load(path, allow_pickle=True)
        self.intrinsic = data["camera_matrix"]
        self.distortion = data["dist_coeffs"]

        mapx, mapy = cv2.initUndistortRectifyMap(self.intrinsic, self.distortion, None, self.intrinsic,
                                                 (self.W, self.H), cv2.CV_32FC1)
        self.mapper[self.W * self.H] = {
            "mapx": mapx,
            "mapy": mapy,
        }

    def read_image_with_calibration(self, imagePath):
        """

        Args:
            imagePath (str):

        Returns:
            ndarray: Undistorted image.
        """
        image = cv2.imread(imagePath)
        w, h = image.shape[1], image.shape[0]

        if w * h in self.mapper.keys():
            mapx = self.mapper[w * h]["mapx"]
            mapy = self.mapper[w * h]["mapy"]
        else:
            mapx, mapy = cv2.initUndistortRectifyMap(self.intrinsic, self.distortion, None, self.intrinsic,
                                                     (w, h), cv2.CV_32FC1)
            self.mapper[w * h] = {
                "mapx": mapx,
                "mapy": mapy,
            }
        calibImage = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR)
        return calibImage

