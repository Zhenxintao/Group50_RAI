import cv2
import numpy as np
import matplotlib.pyplot as plt

from COMP0241_CV.final.utils import get_datasets, ImageReader, load_params
from Task2 import Task2a, Task2b, Task2c
from Task3 import Task3c, Task3e

if __name__ == "__main__":
    calibPath = "calibration_data.npz"
    imageReader = ImageReader(calibPath)

    paths = [
        # "dataset/single_cam/4",
        # "dataset/dual_cam/6/left",
        # "dataset/dual_cam/6/right",
        # "dataset/dual_cam/62/left",
        "dataset/dual_cam/62/right",
        "dataset/dual_cam/G2/left",
        # "dataset/dual_cam/G2/right",
        # "dataset/dual_cam/G1/left",
        # "dataset/dual_cam/G1/right",
    ]
    params = load_params("cfg.json")
    datasets = get_datasets(paths, max_len=400)

    circles = Task2a(datasets, imageReader, display_results=False)
    circles = Task2b(circles, display_results=False)

    # depthList = Task2c(datasets, circles, imageReader, params, display_results=False)  # 2
    # print(depthList)

    # TList = Task3c(datasets, circles, imageReader, method="orb", display_results=False)
    # print(TList)

    TList = Task3e(datasets, circles, imageReader, params, method="orb", display_results=False)
    print(TList)



