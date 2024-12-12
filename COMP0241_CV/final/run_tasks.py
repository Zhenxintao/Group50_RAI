import logging

from COMP0241_CV.final.utils import get_datasets
from Task2 import Task2a, Task2b
from Task3 import Task3c

if __name__ == "__main__":
    paths = [
        # "dataset/dual_cam/6/left",
        "dataset/dual_cam/G1/left",
        "dataset/dual_cam/G1/right",
    ]
    datasets = get_datasets(paths)
    logging.info("Dataset generated")
    logging.debug(datasets[0]["images"][0]["path"])
    logging.debug(datasets[1]["images"][0]["path"])

    circles = Task2a(datasets, max_len=256, display_results=False)
    # Task2b(circles)

    TList = Task3c(datasets, circles, max_len=256, display_results=False)
    print(TList)
