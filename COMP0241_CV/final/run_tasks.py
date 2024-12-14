from utils import get_datasets, ImageReader, load_params
from Task2 import Task2a, Task2b, Task2c
from Task3 import Task3c, Task3e
from Task4 import Task4a, Task4c, Task4d


if __name__ == "__main__":
    calibPath = "calibration_data.npz"
    imageReader = ImageReader(calibPath)

    paths = [
        "dataset/dual_cam/62/left",
        "dataset/dual_cam/62/right",
        # "dataset/dual_cam/G1/left",
        # "dataset/dual_cam/G1/right",
    ]
    # datasets = get_datasets(paths, max_len=50)

    # circles = Task2a(datasets, imageReader, display_results=False)
    # circles = Task2b(circles, display_results=False)

    # depthList = Task2c(datasets, circles, imageReader, (300 + 9.6) / 100., display_results=False)

    params = load_params("cfg.json")
    datasets = get_datasets(paths, max_len=10)

    circles = Task2a(datasets, imageReader, display_results=False)
    circles = Task2b(circles, display_results=False)

    depthList = Task2c(datasets, circles, imageReader, params, display_results=False)  # 2
    # print(depthList)

    # TList = Task3c(datasets, circles, imageReader, method="orb", display_results=False)
    # print(TList)

    TList = Task3e(datasets, circles, imageReader, params, method="orb", display_results=False)
    print(TList)

    # Task4a & Task4b
    real_r = Task4a(circles,imageReader,depthList)

    # Task4c
    Task4c(real_r, TList[0])
    
    # Task4d
    Task4d(datasets, circles, imageReader, method="orb", display_results=False)



    # Task3d(imageReader, "sift", display_results=False)


