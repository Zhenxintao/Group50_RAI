

def get_images_in_folder(path):
    """
    Get all images in a given folder.

    Args:
        path: (str): Path to the image folder.

    Returns:
        List[Dict{
            "name": str,
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
            imageInfo["fps"] = 31.96
        else:
            imageInfo["timestamp"] = int(imageName[:-4])

        imageList.append(imageInfo)
    return imageList


def get_datasets(paths):
    """
    Generate the dataset.

    Args:
        paths (List[str]): folders of images

    Returns:
        List[{
            "path": str,
            "images": List[Dict{
                "name": str,
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
        datasets.append({
            "path": path,
            "images": imageList
        })
    return datasets

