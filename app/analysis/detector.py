import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
from super_gradients.training import models


def get_detector(task):
    """
    Get the appropriate detector based on the specified task.

    Parameters:
    task (str): The task for which the detector is required (e.g., hand movement, leg agility).

    Returns:
    tuple: The detector object and a boolean flag.
    """
    if "hand movement" in str.lower(task):
        return mp_hand(), False
    elif "leg agility" in str.lower(task):
        return yolo_nas_pose(), False
    elif "finger tap" in str.lower(task):
        return mp_hand(), False
    elif "toe tapping" in str.lower(task):
        return test_pose(), False


# Mediapipe pose detector
def mp_pose():
    """
    Create and return a Mediapipe pose detector.

    Returns:
    PoseLandmarker: Mediapipe pose detector.
    """
    VISION_RUNNING_MODE = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path='app/models/pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=VISION_RUNNING_MODE.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options)


def test_pose():
    """
    Create and return a Mediapipe pose detector using the default pose solution.

    Returns:
    Pose: Mediapipe pose detector.
    """
    return mp.solutions.pose.Pose()


# Mediapipe hand detector
def mp_hand():
    """
    Create and return a Mediapipe hand detector.

    Returns:
    HandLandmarker: Mediapipe hand detector.
    """
    VISION_RUNNING_MODE = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path='app/models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=VISION_RUNNING_MODE.VIDEO
    )
    return vision.HandLandmarker.create_from_options(options=options)


# YOLO NAS pose detector
def yolo_nas_pose():
    """
    Create and return a YOLO NAS pose detector.

    Returns:
    YOLOPose: YOLO NAS pose detector model.
    """
    model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else device
    model.to(device)
    return model
