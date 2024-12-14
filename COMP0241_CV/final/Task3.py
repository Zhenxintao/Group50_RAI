import copy
import time

import cv2
import numpy as np

from functions import cal_omega, detect_outliers_1d, detect_largest_circle, cal_omega_any_view, warm_up


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


def Task3c_rotate(datasets, circles, imageReader, method="sift", display_results=True):
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
        imageReader (ImageReader)
        method (str): "sift" or "orb", "sift" has lower time variance,
                                       "orb" has faster average time and better acc.
        display_results (bool)

    Returns:
        List[float]
    """
    if method == "sift":
        matcher = cv2.SIFT_create()
    else:
        matcher = cv2.ORB_create()
        warm_up(datasets, circles, imageReader, matcher)

    TList = []
    timeList = []
    for dataset in datasets:
        omegaList = []
        preImageDict = None
        for k, imageInfoDict in enumerate(dataset["images"]):
            imagePath = imageInfoDict["path"]
            image = imageReader.read_image_with_calibration(imagePath)
            center = circles[dataset["path"]]["centers"][k, :]
            mask = circles[dataset["path"]]["masks"][k, :, :]

            imageDict = {
                "image": image,
                "center": center,
                "mask": mask,
            }
            if "timestamp" in imageInfoDict.keys():
                imageDict["ts"] = imageInfoDict["timestamp"]
            else:
                imageDict["fps"] = imageInfoDict["fps"]

            if preImageDict is None:
                preImageDict = imageDict
                continue

            start = time.time()
            # print(imageDict["center"])
            omega = cal_omega(preImageDict, imageDict, matcher, display_results)
            # print(imageDict["center"])
            timeList.append(time.time() - start)

            preImageDict = imageDict
            omegaList.append(omega)

        omegaList, _ = detect_outliers_1d(np.array(omegaList))
        T = 2 * np.pi / np.mean(omegaList)
        TList.append(T)

    print(f"Average time for one frame: {np.mean((np.array(timeList)))}")
    print(f"Max time for one frame: {np.max(np.array(timeList))}")
    return TList


def Task3e(datasets, circles, imageReader, method="sift", elevation=1, display_results=True):
    if method == "sift":
        matcher = cv2.SIFT_create()
    else:
        matcher = cv2.ORB_create()
        warm_up(datasets, circles, imageReader, matcher)

    TList = []
    for dataset in datasets:
        omegaList = []
        preImageDict = None
        for k, imageInfoDict in enumerate(dataset["images"]):
            imagePath = imageInfoDict["path"]
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

            if elevation == 0:
                omega = cal_omega(preImageDict, imageDict, matcher, display_results)
            else:
                omega = cal_omega_any_view(preImageDict, imageDict, matcher, elevation, display_results)

            preImageDict = imageDict
            omegaList.append(omega)

        omegaList, _ = detect_outliers_1d(np.array(omegaList))
        T = 2 * np.pi / np.mean(omegaList)
        TList.append(T)
    return TList


# def Task3d(imageReader, method, display_results=True):
#     """
#     Real-Time Rotation Cycle Estimation.
#
#     Implement your method to work in real-time, processing live video
#     input (in seconds/rotation).
#
#     Args:
#         imageReader (ImageReader):
#         method (str): "sift" or "ord"
#         display_results (bool)
#     """
#     exit_key = 'q'
#     cap = cv2.VideoCapture(0)
#
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
#
#     if not cap.isOpened():
#         print("Error: Unable to access the camera.")
#         exit()
#
#     print(f"Press '{exit_key}' to quit.")
#
#     i = 0
#     last = 0
#     radiusThreshold = 100
#     distThreshold = 100
#     imageBuffer = [0, 0]
#     historyCircle = [0, 0, 0]
#     while True:
#         start = time.time()
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("Error: Unable to read from the camera.")
#             break
#
#         cv2.imshow("Calibration Capture", frame)
#
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord(exit_key):
#             print("Exiting...")
#             break
#
#         timestamp = start - last
#         image = imageReader.calib_image(frame)
#         circle = detect_largest_circle(image, min_dist=100, param1=50, param2=0.6, display_results=False)
#         if circle is None:
#             circle = historyCircle
#
#         if (
#                 (np.abs(historyCircle[2] - circle[2]) > radiusThreshold) or
#                 (np.abs(historyCircle[0] - circle[0]) + np.abs(historyCircle[1] - circle[1]) > distThreshold)
#             ) and not historyCircle[2] == 0:
#             circle = historyCircle
#
#         historyCircle = circle
#         mask = np.zeros_like(image[:, :, 0])
#         cv2.circle(mask, (circle[0], circle[1]), circle[2], 255, thickness=-1)
#         mask[mask == 255] = 1
#
#         imageDict = {
#             "image": image,
#             "center": circle,
#             "mask": mask,
#             "ts": timestamp
#         }
#
#         if i == 0 or i == 1:
#             imageBuffer[i % 2] = imageDict
#             continue
#
#         prepreImageDict = imageBuffer[i % 2]
#         preImageDict = imageBuffer[(i + 1) % 2]
#         imageBuffer[i % 2] = imageDict
#
#         theta1 = cal_omega(prepreImageDict, preImageDict, method, display_results)
#         theta2 = cal_omega(preImageDict, imageDict, method, display_results)
#
#         theta = (theta1 + theta2) / 2
#         print(theta)
#
#         i += 1
#         last = start
#
#     cap.release()
#     cv2.destroyAllWindows()
