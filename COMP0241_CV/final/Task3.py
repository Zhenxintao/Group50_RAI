import copy

import cv2
import numpy as np

from functions import cal_omega, detect_outliers


def Task3b():
    """
    Provide a Single Rotation Cycle Estimate as reference.

    Calculate the time for one full rotation of the AO. You
    can just mark timestamping on video with manual inspection.

    Returns:
        float: Time for one full rotation.
        float: Average rotation velocity.
    """
    trueT = 51.25 + 7 * 60
    trueOmega = 2 * np.pi / trueT
    return trueT, trueOmega


def Task3c(datasets, circles, imageReader, method="sift", display_results=True):
    """
    Continuous Rotation Cycle Estimation from Video.

    Use video algorithms to automatically estimate the rotation cycle
    over time, noting any variations between 3 captures.
    Args:
        datasets (List[{
            "path": str,
            "images": List[Dict{
                "path": str,
                "frame": int,
                "fps": float,
                "timestamp": int,
            }])
        circles (List[{
            "path": str,
            "circles": ndarray,
        }])
        max_len (int)
        method (str): "sift" or "ord"
        display_results (bool)

    Returns:

    """
    TList = []
    for dataset in datasets:
        thetaList = []
        imageBuffer = [0, 0]
        for k, imageInfoDict in enumerate(dataset["images"]):
            imagePath = imageInfoDict["path"]
            # image = cv2.imread(imagePath)
            image = imageReader.read_image_with_calibration(imagePath)
            center = circles[dataset["path"]]["centers"][k, :]
            mask = circles[dataset["path"]]["masks"][k, :]

            imageDict = {
                "image": image,
                "center": center,
                "mask": mask,
            }
            if "timestamp" in imageInfoDict.keys():
                imageDict["ts"] = imageInfoDict["timestamp"]
            else:
                imageDict["fps"] = imageInfoDict["fps"]

            if k == 0 or k == 1:
                imageBuffer[k % 2] = imageDict
                continue

            prepreImageDict = imageBuffer[k % 2]
            preImageDict = imageBuffer[(k + 1) % 2]
            imageBuffer[k % 2] = imageDict

            theta1 = cal_omega(prepreImageDict, preImageDict, method, display_results)
            theta2 = cal_omega(preImageDict, imageDict, method, display_results)

            theta = (theta1 + theta2) / 2
            thetaList.append(theta)
        thetaList = np.array(thetaList)
        thetaList, _ = detect_outliers(thetaList)
        print(np.mean(thetaList))
        T = 2 * np.pi / np.mean(thetaList)
        print(T)
        TList.append(T)
    return TList



