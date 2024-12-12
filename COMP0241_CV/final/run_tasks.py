import logging

from COMP0241_CV.final.utils import get_datasets, ImageReader
from Task2 import Task2a, Task2b, Task2c
from Task3 import Task3c

if __name__ == "__main__":
    calibPath = "calibration_data.npz"
    imageReader = ImageReader(calibPath)

    paths = [
        # "dataset/dual_cam/6/left",
        "dataset/dual_cam/G1/left",
        "dataset/dual_cam/G1/right",
    ]
    datasets = get_datasets(paths, max_len=256)

    circles = Task2a(datasets, imageReader, display_results=False)
    # Task2b(circles)
    Task2c(datasets, circles, imageReader, display_results=False)

    # TList = Task3c(datasets, circles, imageReader, display_results=False)
    # print(TList)
