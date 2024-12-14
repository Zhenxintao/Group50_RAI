import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from functions import warm_up, cal_omega


def get_radius_meters(img_radius, fy, depth):
    """
    Gets the true radius length

    Args:
        img_radius: The pixel value of the radius in the picture
        fy: Internal parameter matrix
        depth: Actual depth
    """
    r = (-img_radius * depth) / (img_radius - fy)
    return r


# Task4a & Task4b
def Task4a(circles, imageReader, depth_List):
    """
    Args:
        circles (List[{
            "path": str,
            "circles": ndarray,
        }])
        imageReader (ImageReader)
        depth_List (List[float])
    """
    fy = imageReader.fy
    real_r_list = []
    
    for path, circle in circles.items():
        if path == "final/dataset/dual_cam/62/left":
            img_radius_list = circle["centers"][:, 2]  
            for i, img_radius in enumerate(img_radius_list):
                depth = depth_List[0][i]  
                real_r = get_radius_meters(img_radius, fy, depth)
                real_r_list.append(real_r)

    average = np.mean(real_r_list)
    diameter = average * 2
    print(f"Diameter:{diameter}")

    # Task 4b: Visualization of real-world radii differences
    plt.figure(figsize=(10, 5)) 

    plt.plot(real_r_list, label="Calculated Radii (meters)", marker='o')

    plt.title(f"The Linear Velocity Of AO \n(Diameter = {diameter:.2f} meters)", fontsize=14)
    plt.xlabel("Index of Images", fontsize=12)  
    plt.ylabel("Radius in Meters", fontsize=12)  

    plt.text(len(real_r_list) - 1, max(real_r_list), f"Diameter = {diameter:.2f} m", 
             fontsize=12, color="green", ha="right", va="bottom", bbox=dict(facecolor='white', alpha=0.6))

    plt.grid(alpha=0.3)
    plt.legend()

    plt.show()

    return average


def Task4c(radius_meters, rotation_period):
    """
    Calculate the surface linear velocity

    Args:
         radius_meters: Radius of a sphere (in meters)
         rotation_period: rotation period
    """
    R = radius_meters 
    T = rotation_period 
    omega = 2 * np.pi / T  

    latitudes = np.linspace(-90, 90, 1000)
    latitudes_rad = np.radians(latitudes) 

    linear_velocity = omega * R * np.cos(latitudes_rad)

    plt.figure(figsize=(12, 8))
    
    plt.plot(latitudes, linear_velocity, label="Surface Linear Velocity (m/s)", color="blue", linewidth=2)
    
    critical_latitudes = [-90, -45, 0, 45, 90]
    critical_velocities = omega * R * np.cos(np.radians(critical_latitudes))
    for lat, vel in zip(critical_latitudes, critical_velocities):
        plt.scatter(lat, vel, color='red', zorder=5) 
        plt.text(lat, vel + 20, f"{vel:.2f} m/s", fontsize=10, ha='center', color='red')  

    plt.axhline(0, color='black', linewidth=1, linestyle='--', label="Zero Velocity (at poles)")
    plt.grid(alpha=0.3, linestyle='--')

    plt.title("Relation Between Surface Linear Velocity and Latitude", fontsize=16)
    plt.xlabel("Latitude (°)", fontsize=14)
    plt.ylabel("Linear Velocity (m/s)", fontsize=14)
    
    plt.legend(fontsize=12, loc="upper right")
    
    plt.tight_layout()
    plt.show()



def Task4d(datasets, circles, imageReader, method="sift", display_results=True):
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

    timeList = []
    for dataset in datasets:
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

            if preImageDict is not None:
                start = time.time()
                omega = cal_omega(preImageDict, imageDict, matcher, display_results=display_results)
                R = center[2]
                linerVelocity = omega * R
                timeList.append(time.time() - start)
            preImageDict = imageDict

    print(f"Average time for one frame: {np.mean((np.array(timeList)))}")
    print(f"Max time for one frame: {np.max(np.array(timeList))}")





