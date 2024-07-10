import math
import numpy as np
import mediapipe as mp
import cv2

import app.analysis.constants.mp_handlandmarks as MP_HAND_LANDMARKS
import app.analysis.constants.mp_landmarks as MP_LANDMARKS
import app.analysis.constants.yolo_landmarks as YOLO_LANDMARKS

from mediapipe.framework.formats import landmark_pb2

def draw_opt(rgb_image, detection_result):
    """
    Draw pose landmarks on the image.

    Parameters:
    rgb_image (numpy array): The input RGB image.
    detection_result: Detection result containing pose landmarks.

    Returns:
    numpy array: Annotated image with pose landmarks.
    """
    pose_landmarks_list = detection_result.pose_landmarks[0]
    annotated_image = np.copy(rgb_image)
    
    # Create a NormalizedLandmarkList for pose landmarks
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks_list
    ])
    
    # Draw the pose landmarks on the annotated image
    mp.solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        mp.solutions.pose.POSE_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    
    return annotated_image

def draw_hand(rgb_image, hand_landmarks, bounds=None):
    """
    Draw hand landmarks on the image.

    Parameters:
    rgb_image (numpy array): The input RGB image.
    hand_landmarks: List of hand landmarks.
    bounds (list, optional): Bounding box for the hand region.

    Returns:
    numpy array: Annotated image with hand landmarks.
    """
    annotated_image = np.copy(rgb_image)
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

    try:
        # Normalize and add landmarks to the list
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
    except:
        # If bounds are provided, normalize landmarks with respect to the bounding box
        [x1, y1, x2, y2] = bounds
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark[0] / (x2 - x1), y=landmark[1] / (y2 - y1), z=landmark[2]) for landmark in hand_landmarks
        ])
    
    # Draw the hand landmarks on the annotated image
    mp.solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto
    )
    
    return annotated_image

def get_essential_landmarks(current_frame, current_frame_idx, task, bounding_box, detector):
    """
    Get essential landmarks based on the specified task.

    Parameters:
    current_frame (numpy array): The current frame of the video.
    current_frame_idx (int): The index of the current frame.
    task (str): The task being performed (e.g., hand movement, finger tap).
    bounding_box (dict): The bounding box for the region of interest.
    detector: The detector object for detecting landmarks.

    Returns:
    list: List of essential landmarks for the specified task.
    """
    is_left = False
    if "left" in str.lower(task):
        is_left = True

    if "hand movement" in str.lower(task):
        return get_hand_movement_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector)
    elif "finger tap" in str.lower(task):
        return get_finger_tap_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector)
    elif "leg agility" in str.lower(task):
        return get_leg_agility_landmarks(bounding_box, detector, current_frame, current_frame_idx, is_left)
    elif "toe tapping" in str.lower(task):
        return get_toe_tapping_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left)

def get_signal(display_landmarks, task):
    """
    Get signal data based on the specified task.

    Parameters:
    display_landmarks (list): List of display landmarks.
    task (str): The task being performed (e.g., hand movement, finger tap).

    Returns:
    list: Signal data for the specified task.
    """
    if "hand movement" in str.lower(task):
        return get_hand_movement_signal(display_landmarks)
    elif "finger tap" in str.lower(task):
        return get_finger_tap_signal(display_landmarks)
    elif "leg agility" in str.lower(task):
        return get_leg_agility_signal(display_landmarks)
    elif "toe tapping" in str.lower(task):
        return get_toe_tapping_signal(display_landmarks)

def get_normalisation_factor(landmarks, task):
    """
    Get normalization factor based on the specified task.

    Parameters:
    landmarks (list): List of landmarks.
    task (str): The task being performed (e.g., hand movement, finger tap).

    Returns:
    float: Normalization factor for the specified task.
    """
    if "hand movement" in str.lower(task):
        return get_hand_movement_nf(landmarks)
    elif "finger tap" in str.lower(task):
        return get_finger_tap_nf(landmarks)
    elif "leg agility" in str.lower(task):
        return get_leg_agility_nf(landmarks)
    elif "toe tapping" in str.lower(task):
        return get_toe_tapping_nf(landmarks)

def get_display_landmarks(landmarks, task):
    """
    Get display landmarks based on the specified task.

    Parameters:
    landmarks (list): List of landmarks.
    task (str): The task being performed (e.g., hand movement, finger tap).

    Returns:
    list: Display landmarks for the specified task.
    """
    if "hand movement" in str.lower(task):
        return get_hand_movement_display_landmarks(landmarks)
    elif "finger tap" in str.lower(task):
        return get_finger_tap_display_landmarks(landmarks)
    elif "leg agility" in str.lower(task):
        return get_leg_agility_display_landmarks(landmarks)
    elif "toe tapping" in str.lower(task):
        return get_toe_tapping_display_landmarks(landmarks)

def get_leg_agility_landmarks(bounding_box, detector, current_frame, current_frame_idx, is_left):
    """
    Get leg agility landmarks from the bounding box and current frame.

    Parameters:
    bounding_box (dict): The bounding box for the region of interest.
    detector: The detector object for detecting landmarks.
    current_frame (numpy array): The current frame of the video.
    current_frame_idx (int): The index of the current frame.
    is_left (bool): Whether to get landmarks for the left leg.

    Returns:
    list: List of leg agility landmarks.
    """
    [x1, y1, x2, y2] = get_boundaries(bounding_box)
    roi = current_frame[y1:y2, x1:x2]

    # Convert the ROI to RGB since many models expect input in this format
    results = detector.predict(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), conf=0.7)
    landmarks = results.prediction.poses[0]

    knee_idx = YOLO_LANDMARKS.LEFT_KNEE if is_left else YOLO_LANDMARKS.RIGHT_KNEE

    left_shoulder = landmarks[YOLO_LANDMARKS.LEFT_SHOULDER]
    right_shoulder = landmarks[YOLO_LANDMARKS.RIGHT_SHOULDER]
    knee_landmark = landmarks[knee_idx]
    left_hip = landmarks[YOLO_LANDMARKS.LEFT_HIP]
    right_hip = landmarks[YOLO_LANDMARKS.RIGHT_HIP]

    return [left_shoulder, right_shoulder, knee_landmark, left_hip, right_hip]

def get_leg_agility_signal(landmarks_list):
    """
    Get leg agility signal from the landmarks list.

    Parameters:
    landmarks_list (list): List of leg agility landmarks.

    Returns:
    list: Leg agility signal data.
    """
    signal = []
    for landmarks in landmarks_list:
        [shoulder_midpoint, knee_landmark] = landmarks
        distance = math.dist(knee_landmark[:2], shoulder_midpoint)
        signal.append(distance)
    return signal

def get_leg_agility_nf(landmarks_list):
    """
    Get leg agility normalization factor from the landmarks list.

    Parameters:
    landmarks_list (list): List of leg agility landmarks.

    Returns:
    float: Leg agility normalization factor.
    """
    values = []
    for landmarks in landmarks_list:
        [left_shoulder, right_shoulder, _, left_hip, right_hip] = landmarks
        shoulder_midpoint = (np.array(left_shoulder[:2]) + np.array(right_shoulder[:2])) / 2
        hip_midpoint = (np.array(left_hip[:2]) + np.array(right_hip[:2])) / 2
        distance = math.dist(shoulder_midpoint, hip_midpoint)
        values.append(distance)
    return np.mean(values)

def get_leg_agility_display_landmarks(landmarks_list):
    """
    Get leg agility display landmarks from the landmarks list.

    Parameters:
    landmarks_list (list): List of leg agility landmarks.

    Returns:
    list: Display landmarks for leg agility.
    """
    display_landmarks = []
    for landmarks in landmarks_list:
        [left_shoulder, right_shoulder, knee_landmark, _, _] = landmarks
        shoulder_midpoint = (np.array(left_shoulder[:2]) + np.array(right_shoulder[:2])) / 2
        display_landmarks.append([shoulder_midpoint, knee_landmark])
    return display_landmarks

def get_toe_tapping_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left):
    """
    Get toe tapping landmarks from the bounding box and current frame.

    Parameters:
    bounding_box (dict): The bounding box for the region of interest.
    detector: The detector object for detecting landmarks.
    current_frame (numpy array): The current frame of the video.
    current_frame_idx (int): The index of the current frame.
    is_left (bool): Whether to get landmarks for the left foot.

    Returns:
    list: List of toe tapping landmarks.
    """
    [x1, y1, x2, y2] = get_boundaries(bounding_box)
    frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

    landmarks = detector.process(frame[y1:y2, x1:x2, :]).pose_landmarks.landmark

    knee_idx = MP_LANDMARKS.LEFT_KNEE if is_left else MP_LANDMARKS.RIGHT_KNEE
    toe_idx = MP_LANDMARKS.LEFT_FOOT_INDEX if is_left else MP_LANDMARKS.RIGHT_FOOT_INDEX

    left_shoulder = landmarks[MP_LANDMARKS.LEFT_SHOULDER]
    right_shoulder = landmarks[MP_LANDMARKS.RIGHT_SHOULDER]
    shoulder_midpoint = [(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2]
    shoulder_midpoint = [shoulder_midpoint[0] * (x2 - x1), shoulder_midpoint[1] * (y2 - y1)]

    left_hip = landmarks[MP_LANDMARKS.LEFT_HIP]
    right_hip = landmarks[MP_LANDMARKS.RIGHT_HIP]
    hip_midpoint = [(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2]
    hip_midpoint = [hip_midpoint[0] * (x2 - x1), hip_midpoint[1] * (y2 - y1)]

    toe_landmark = [landmarks[toe_idx].x * (x2 - x1), landmarks[toe_idx].y * (y2 - y1)]

    return [shoulder_midpoint, toe_landmark, hip_midpoint]

def get_toe_tapping_signal(landmarks_list):
    """
    Get toe tapping signal from the landmarks list.

    Parameters:
    landmarks_list (list): List of toe tapping landmarks.

    Returns:
    list: Toe tapping signal data.
    """
    signal = []
    for landmarks in landmarks_list:
        [shoulder_midpoint, toe] = landmarks
        distance = math.dist(shoulder_midpoint, toe)
        signal.append(distance)
    return signal

def get_toe_tapping_nf(landmarks_list):
    """
    Get toe tapping normalization factor from the landmarks list.

    Parameters:
    landmarks_list (list): List of toe tapping landmarks.

    Returns:
    float: Toe tapping normalization factor.
    """
    values = []
    for landmarks in landmarks_list:
        [shoulder_midpoint, _, hip_midpoint] = landmarks
        distance = math.dist(shoulder_midpoint, hip_midpoint)
        values.append(distance)
    return np.mean(values)

def get_toe_tapping_display_landmarks(landmarks_list):
    """
    Get toe tapping display landmarks from the landmarks list.

    Parameters:
    landmarks_list (list): List of toe tapping landmarks.

    Returns:
    list: Display landmarks for toe tapping.
    """
    display_landmarks = []
    for landmarks in landmarks_list:
        [shoulder_midpoint, toe_landmark, _] = landmarks
        display_landmarks.append([shoulder_midpoint, toe_landmark])
    return display_landmarks

def get_hand_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left):
    """
    Get hand landmarks from the bounding box and current frame.

    Parameters:
    bounding_box (dict): The bounding box for the region of interest.
    detector: The detector object for detecting landmarks.
    current_frame (numpy array): The current frame of the video.
    current_frame_idx (int): The index of the current frame.
    is_left (bool): Whether to get landmarks for the left hand.

    Returns:
    list: List of hand landmarks.
    """
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    [x1, y1, x2, y2] = get_boundaries(bounding_box)

    image_data = current_frame[y1:y2, x1:x2, :].astype(np.uint8)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)

    detection_result = detector.detect_for_video(image, current_frame_idx)
    current_frame_idx += 1

    hand_index = get_hand_index(detection_result, is_left)

    if hand_index == -1:
        return []  # Skip frame if hand is not detected

    return detection_result.hand_landmarks[hand_index]

def get_hand_movement_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector):
    """
    Get hand movement landmarks from the bounding box and current frame.

    Parameters:
    current_frame (numpy array): The current frame of the video.
    current_frame_idx (int): The index of the current frame.
    bounding_box (dict): The bounding box for the region of interest.
    is_left (bool): Whether to get landmarks for the left hand.
    detector: The detector object for detecting landmarks.

    Returns:
    list: List of hand movement landmarks.
    """
    hand_landmarks = get_hand_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left)
    if not hand_landmarks:
        return []
    bounds = get_boundaries(bounding_box)
    index_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP], bounds)
    middle_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.MIDDLE_FINGER_TIP], bounds)
    ring_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.RING_FINGER_TIP], bounds)
    wrist = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.WRIST], bounds)

    if current_frame_idx == 6667:
        [x1, y1, x2, y2] = bounds
        landmarks = [hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP], hand_landmarks[MP_HAND_LANDMARKS.MIDDLE_FINGER_TIP], hand_landmarks[MP_HAND_LANDMARKS.RING_FINGER_TIP]]
        cv2.imwrite("outputs/" + str(current_frame_idx) + ".jpg", draw_hand(current_frame[y1:y2, x1:x2, :], landmarks))

    return [index_finger, middle_finger, ring_finger, wrist]

def get_hand_movement_signal(landmarks_list):
    """
    Get hand movement signal from the landmarks list.

    Parameters:
    landmarks_list (list): List of hand movement landmarks.

    Returns:
    list: Hand movement signal data.
    """
    signal = []
    prevDist = 0
    for landmarks in landmarks_list:
        if len(landmarks) < 4:
            signal.append(prevDist)
            continue
        [index_finger, middle_finger, ring_finger, wrist] = landmarks
        distance = (math.dist(index_finger, wrist) + math.dist(middle_finger, wrist) + math.dist(ring_finger, wrist)) / 3
        prevDist = distance
        signal.append(distance)
    return signal

def get_hand_movement_nf(landmarks_list):
    """
    Get hand movement normalization factor from the landmarks list.

    Parameters:
    landmarks_list (list): List of hand movement landmarks.

    Returns:
    float: Hand movement normalization factor.
    """
    values = []
    for landmarks in landmarks_list:
        if len(landmarks) < 4:
            continue  # Skip this iteration if landmarks length is less than 4
        [_, middle_finger, _, wrist] = landmarks
        distance = math.dist(middle_finger, wrist)
        values.append(distance)
    return np.max(values)

def get_hand_movement_display_landmarks(landmarks_list):
    """
    Get hand movement display landmarks from the landmarks list.

    Parameters:
    landmarks_list (list): List of hand movement landmarks.

    Returns:
    list: Display landmarks for hand movement.
    """
    display_landmarks = []
    for landmarks in landmarks_list:
        if len(landmarks) < 4:
            display_landmarks.append([])
            continue  # Skip this iteration if landmarks length is less than 4
        [index_finger, middle_finger, ring_finger, wrist] = landmarks
        display_landmarks.append([index_finger, middle_finger, ring_finger, wrist])
    return display_landmarks

def get_finger_tap_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector):
    """
    Get finger tap landmarks from the bounding box and current frame.

    Parameters:
    current_frame (numpy array): The current frame of the video.
    current_frame_idx (int): The index of the current frame.
    bounding_box (dict): The bounding box for the region of interest.
    is_left (bool): Whether to get landmarks for the left hand.
    detector: The detector object for detecting landmarks.

    Returns:
    list: List of finger tap landmarks.
    """
    hand_landmarks = get_hand_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left)
    if not hand_landmarks:
        return []

    bounds = get_boundaries(bounding_box)
    thumb_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.THUMB_TIP], bounds)
    index_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP], bounds)
    middle_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.MIDDLE_FINGER_TIP], bounds)
    wrist = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.WRIST], bounds)

    if current_frame_idx == 1408:
        [x1, y1, x2, y2] = bounds
        landmarks = [hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP], hand_landmarks[MP_HAND_LANDMARKS.THUMB_TIP]]
        cv2.imwrite("outputs/" + str(current_frame_idx) + ".jpg", draw_hand(current_frame[y1:y2, x1:x2, :], landmarks))

    return [thumb_finger, index_finger, middle_finger, wrist]

def get_finger_tap_signal(landmarks_list):
    """
    Get finger tap signal from the landmarks list.

    Parameters:
    landmarks_list (list): List of finger tap landmarks.

    Returns:
    list: Finger tap signal data.
    """
    signal = []
    prev = 0
    for landmarks in landmarks_list:
        if not landmarks:
            signal.append(prev)
            continue
        [thumb_finger, index_finger] = landmarks
        distance = math.dist(thumb_finger, index_finger)
        prev = distance
        signal.append(distance)
    return signal

def get_finger_tap_nf(landmarks_list):
    """
    Get finger tap normalization factor from the landmarks list.

    Parameters:
    landmarks_list (list): List of finger tap landmarks.

    Returns:
    float: Finger tap normalization factor.
    """
    values = []
    for landmarks in landmarks_list:
        if not landmarks:
            continue
        [_, _, middle_finger, wrist] = landmarks
        distance = math.dist(middle_finger, wrist)
        values.append(distance)
    return np.max(values)

def get_finger_tap_display_landmarks(landmarks_list):
    """
    Get finger tap display landmarks from the landmarks list.

    Parameters:
    landmarks_list (list): List of finger tap landmarks.

    Returns:
    list: Display landmarks for finger tap.
    """
    display_landmarks = []
    for landmarks in landmarks_list:
        if not landmarks:
            display_landmarks.append([])
            continue
        [thumb_finger, index_finger, _, _] = landmarks
        display_landmarks.append([thumb_finger, index_finger])
    return display_landmarks

def get_hand_index(detection_result, is_left):
    """
    Get the index of the detected hand (left or right) from the detection result.

    Parameters:
    detection_result: Detection result containing hand landmarks.
    is_left (bool): Whether to get the index for the left hand.

    Returns:
    int: Index of the detected hand, or -1 if not found.
    """
    direction = "Left" if is_left else "Right"
    handedness = detection_result.handedness

    for idx in range(0, len(handedness)):
        if handedness[idx][0].category_name == direction:
            return idx

    return -1

def get_landmark_coords(landmark, bounds):
    """
    Get the coordinates of a landmark within the bounding box.

    Parameters:
    landmark: A single landmark.
    bounds (list): The bounding box coordinates.

    Returns:
    list: The normalized coordinates of the landmark.
    """
    [x1, y1, x2, y2] = bounds
    return [landmark.x * (x2 - x1), landmark.y * (y2 - y1)]

def get_boundaries(bounding_box):
    """
    Get the boundaries of the bounding box.

    Parameters:
    bounding_box (dict): The bounding box for the region of interest.

    Returns:
    list: The coordinates of the bounding box.
    """
    x1 = int(bounding_box['x'])
    y1 = int(bounding_box['y'])
    x2 = x1 + int(bounding_box['width'])
    y2 = y1 + int(bounding_box['height'])

    return [x1, y1, x2, y2]
