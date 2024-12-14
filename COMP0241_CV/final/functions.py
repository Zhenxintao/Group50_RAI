import copy
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_largest_circle(image, min_dist=500, param1=100, param2=20, min_radius=0, max_radius=0,
                          display_results=True):
    """
    Detects the largest circle in an image and returns its radius.

    Args:
        image (str): The image file.
        min_dist (int): Minimum distance between the centers of detected circles.
        param1 (int): First method-specific parameter for cv2.HoughCircles.
        param2 (int): Second method-specific parameter for cv2.HoughCircles.
        min_radius (int): Minimum circle radius.
        max_radius (int): Maximum circle radius.
        display_results (bool): Whether to display the detected circle and binary mask.

    Returns:
        int: The radius of the largest circle if detected, otherwise None.
    """
    image = copy.copy(image)
    gray = copy.copy(image[:, :, 1])
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.2,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        circles = np.int16(np.around(circles))
        largest_circle = max(circles[0, :], key=lambda x: x[2])
        mask = np.zeros_like(gray)
        # largest_circle[0], largest_circle[1] = bestCircle[0], bestCircle[1]
        cv2.circle(mask, (largest_circle[0], largest_circle[1]), largest_circle[2], 255, thickness=-1)
        mask[mask == 255] = 1

        if display_results:
            # cv2.circle(image, (bestCircle[0], bestCircle[1]), largest_circle[2], (0, 255, 0), 4)
            cv2.circle(image, (largest_circle[0], largest_circle[1]), largest_circle[2], (0, 0, 255), 4)
            cv2.circle(image, (largest_circle[0], largest_circle[1]), 2, (0, 0, 255), 3)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Detected Circle")
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("Binary Mask of Circle")
            plt.imshow(mask, cmap="gray")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        return largest_circle
    else:
        print("No circles detected.")
        return None


def kalman_filter(observations, A, H, Q, R, P0):
    n = len(observations)
    estimates = np.zeros(n)

    x = observations[0]
    P = P0

    for i in range(n):
        x_pred = A * x
        P_pred = A * P * A + Q

        K = P_pred * H / (H * P_pred * H + R)
        x = x_pred + K * (observations[i] - H * x_pred)
        P = (1 - K * H) * P_pred

        estimates[i] = x
    return estimates


def detect_outliers_1d(data, threshold=3):
    """
    Detect the outliers and filter the data.

    Args:
        data (ndarray): [N]
        threshold (int)

    Returns:
        ndarray: filtered data
        ndarray: outliers
    """
    mean = np.mean(data)
    std_dev = np.std(data)

    outliers = []
    filteredData = []
    for value in data:
        if abs(value - mean) > threshold * std_dev:
            outliers.append(value)
        else:
            filteredData.append(value)
    return np.array(filteredData), np.array(outliers)


def detect_outliers_2d(data, threshold=3):
    """
    Detect the outliers and filter the data.

    Args:
        data (ndarray): [N, 2]
        threshold (int)

    Returns:
        ndarray: filtered data
        ndarray: outliers
    """
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)

    outMaskX = np.abs(data[:, 0] - mean[0]) > threshold * std_dev[0]
    outMaskY = np.abs(data[:, 1] - mean[1]) > threshold * std_dev[1]
    outMask = np.logical_or(outMaskX, outMaskY)

    outliers = data[outMask]
    filteredData = data[~outMask]
    return np.array(filteredData), np.array(outliers)


def get_match_pts(imgDict1, imgDict2, matcher, display_results=True):
    """
    Get match points between 2 images with a mask
    to control the matching range

    Args:
        imgDict1 (Dict{
                "image": ndarray,
                "center": ndarray,
                "mask": ndarray,
            }): Information about the first image.
        imgDict2 (Dict{
                "image": ndarray,
                "center": ndarray,
                "mask": ndarray,
            }): Information about the second image.
        matcher (cv2.SIFT | cv2.ORB):
        display_results (bool)

    Returns:
        ndarray(np.float32): Key point pairs in the first image. [N, 2, 2]
    """
    img1, img2 = imgDict1["image"], imgDict2["image"]

    kpts1, descriptors1 = matcher.detectAndCompute(img1, imgDict1["mask"])
    kpts2, descriptors2 = matcher.detectAndCompute(img2, imgDict2["mask"])

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    if display_results:
        matched_img = cv2.drawMatches(
            img1, kpts1, img2, kpts2, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        scale_percent = 50
        new_width = int(matched_img.shape[1] * scale_percent / 100)
        new_height = int(matched_img.shape[0] * scale_percent / 100)
        dim = (new_width, new_height)

        resized_img = cv2.resize(matched_img, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow("Keypoint Matches", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    kptsPairs = [[kpts1[m.queryIdx].pt, kpts2[m.trainIdx].pt] for m in matches]
    return np.array(kptsPairs, dtype=np.float32)


def crop_two_img(imgDict1, imgDict2):
    """
    Crop two frame by finding the minimum bound.

    Args:
        imgDict1:
        imgDict2:

    Returns:
        Dict{
            "image": ndarray,
            "center": ndarray,
            "mask": ndarray,
        }: Cropped imgDict1.
        Dict{
            "image": ndarray,
            "center": ndarray,
            "mask": ndarray,
        }: Cropped imgDict2.
    """
    imgDict1, imgDict2 = copy.deepcopy(imgDict1), copy.deepcopy(imgDict2)
    c1 = imgDict1["center"]
    c2 = imgDict2["center"]

    H, W = imgDict1["image"].shape[:2]
    wBound1 = min(c1[0] - c1[2], W - c1[0] - c1[2])
    wBound2 = min(c2[0] - c2[2], W - c2[0] - c2[2])
    wb = int(min(wBound1, wBound2))

    hBound1 = min(c1[1] - c1[2], H - c1[1] - c1[2])
    hBound2 = min(c2[1] - c2[2], H - c2[1] - c2[2])
    hb = int(min(hBound1, hBound2))

    imgDict1["image"] = imgDict1["image"][hb:-hb, wb:-wb, :]
    imgDict2["image"] = imgDict2["image"][hb:-hb, wb:-wb, :]
    imgDict1["mask"] = imgDict1["mask"][hb:-hb, wb:-wb]
    imgDict2["mask"] = imgDict2["mask"][hb:-hb, wb:-wb]
    imgDict1["center"][:2] -= [wb, hb]
    imgDict2["center"][:2] -= [wb, hb]
    return imgDict1, imgDict2


def get_affine_mat(imgDict1, imgDict2, kptsPairs, display_results=True):
    """
    Calculate rotation matrix by matched key points and cv2.estimateAffinePartial2D

    Args:
        imgDict1:
        imgDict2:
        kptsPairs (ndarray): Key point pairs in the first image. [N, 2, 2]
        display_results (bool)

    Returns:
        ndarray: Affine matrix. [2, 3]
    """
    c1 = imgDict1["center"]
    c2 = imgDict2["center"]
    kpts1 = kptsPairs[:, 0, :] - c1[:2]
    kpts2 = kptsPairs[:, 1, :] - c2[:2]
    src = kpts1[:, np.newaxis, :]
    trg = kpts2[:, np.newaxis, :]

    matrix, _ = cv2.estimateAffinePartial2D(src, trg)
    if display_results:
        kpts3 = cv2.transform(np.array([kpts1]), matrix)[0]
        img1 = copy.copy(imgDict1["image"])
        img2 = copy.copy(imgDict2["image"])
        canvas = img1 // 2 + img2 // 2
        kpts1 = kpts1 + c1[:2]
        kpts2 = kpts2 + c2[:2]
        kpts3 = kpts3 + c2[:2]

        for point in kpts1: cv2.circle(canvas, tuple(point.astype(int)), 4, (0, 255, 0), -1)
        for point in kpts2: cv2.circle(canvas, tuple(point.astype(int)), 3, (0, 0, 255), -1)
        for point in kpts3: cv2.circle(canvas, tuple(point.astype(int)), 2, (255, 0, 0), -1)

        for p1, p2 in zip(kpts1, kpts3):
            cv2.line(canvas, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 255, 255), 1)

        cv2.imshow("Affine Transform Visualization", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return matrix


def get_equator_mask(imgDict, pts, elevation, offset, inverse):
    """
    Find the mask represents the region nearby the equator.

    Args:
        imgDict:
        pts (ndarray): [N, 2]
        elevation:
        offset (int): The maximum deviation above and below the equator.
        inverse (bool): Inverse the coordinate.

    Returns:
        ndarray
    """
    r = imgDict["center"][2]
    H = imgDict["image"].shape[0]
    if inverse:
        elevation = 90 - elevation
        equatorH = H - int(r * np.sin(np.radians(elevation)))
    else:
        equatorH = r - int(r * np.sin(np.radians(elevation)))

    equatorMaskAbove = pts[:, 1] < equatorH + offset
    equatorMaskBelow = pts[:, 1] > equatorH - offset
    return np.logical_and(equatorMaskAbove, equatorMaskBelow)


def get_pts_in_equator(imgDict1, imgDict2, kptsPairs, elevation, offset=20, inverse=False):
    """

    Args:
        imgDict1:
        imgDict2:
        kptsPairs (ndarray): [N, 2, 2]
        elevation:
        offset (int): The maximum deviation above and below the equator.
        inverse (bool): Inverse the coordinate

    Returns:
        ndarray: Key point pairs nearby the equator.
    """
    equatorMask1 = get_equator_mask(imgDict1, kptsPairs[:, 0, :], elevation, offset, inverse)
    equatorMask2 = get_equator_mask(imgDict2, kptsPairs[:, 1, :], elevation, offset, inverse)
    equatorMask = np.logical_and(equatorMask1, equatorMask2)
    return kptsPairs[equatorMask]


def warm_up(datasets, circles, imageReader, matcher):
    """
    Run the matcher.detectAndCompute() function in cal_omega() in advance
    to address the time-consuming issue during the first execution.

    Args:
        datasets:
        circles:
        imageReader:
        matcher:
    """
    imgP = datasets[0]["images"][0]["path"]
    image = imageReader.read_image_with_calibration(imgP)
    center = circles[datasets[0]["path"]]["centers"][0, :]
    mask = circles[datasets[0]["path"]]["masks"][0, :]

    imgD = {
        "image": image,
        "center": center,
        "mask": mask,
    }
    if "timestamp" in datasets[0]["images"][0].keys():
        imgD["ts"] = datasets[0]["images"][0]["timestamp"]
    else:
        imgD["fps"] = datasets[0]["images"][0]["fps"]
    imgD = copy.deepcopy(imgD)
    cal_omega(imgD, imgD, matcher, elevation=90, display_results=False)


def cal_theta_vertical(imgD1, imgD2, kptsPairs, display_results=True):
    """
    Calculate the OA's rotation velocity by given two image in vertical view.
    """
    matrix = get_affine_mat(imgD1, imgD2, kptsPairs, display_results)
    rotMat = matrix[:, :2]
    theta = np.arctan2(rotMat[1, 0], rotMat[0, 0])
    return theta


def cal_theta_inclination(imgD1, imgD2, kptsPairs, elevation, offset=20, inverse=False, display_results=True):
    """
    Calculate the OA's rotation velocity by given two image from any view.
    """
    eqPairs = get_pts_in_equator(imgD1, imgD2, kptsPairs, elevation, offset, inverse)
    matrix = get_affine_mat(imgD1, imgD2, eqPairs, display_results)
    deltaX = matrix[0, -1]
    r = (imgD1["center"][2] + imgD2["center"][2]) / 2
    theta = np.arcsin(deltaX / r)
    return theta


def cal_omega(imgDict1, imgDict2, matcher, elevation=0, offset=20, interval=10, inverse=False, display_results=True):
    """
    Calculate the OA's rotation velocity by given two image.

    Args:
        imgDict1 (Dict{
                "image": ndarray,
                "center": ndarray,
                "mask": ndarray,
            }): Information about the first image.
        imgDict2 (Dict{
                "image": ndarray,
                "center": ndarray,
                "mask": ndarray,
            }): Information about the second image.
        matcher (cv2.SIFT | cv2.ORB):
        elevation (float): 90 is vertical, 0 is horizontal.
        offset (int): The maximum deviation above and below the equator.
        interval (int): The interval number of frames.
        inverse (bool): Inverse the coordinate.
        display_results (bool)

    Returns:
        float: Rotation velocity.
    """
    imgD1, imgD2 = crop_two_img(imgDict1, imgDict2)
    kptsPairs = get_match_pts(imgD1, imgD2, matcher, display_results)
    if elevation == 90:
        theta = cal_theta_vertical(imgD1, imgD2, kptsPairs, display_results)
    else:
        theta = cal_theta_inclination(imgD1, imgD2, kptsPairs, elevation, offset, inverse, display_results)
    if "ts" in imgD1.keys(): t = (imgD2["ts"] - imgD1["ts"]) / 1e6
    else:                    t = 1. / imgD1["fps"]
    t *= interval
    return theta / (t + 1e-9)
