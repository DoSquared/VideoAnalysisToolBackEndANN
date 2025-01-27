import math
import numpy as np
import cv2
from app.analysis.util import filter_signal, get_output
from app.analysis.detector import get_detector
from app.analysis.task_analysis import get_essential_landmarks, get_signal, get_normalisation_factor, get_display_landmarks
import scipy.signal as signal

def analysis(bounding_box, start_time, end_time, input_video, task_name):
    """
    Perform analysis on the specified task in the given video.

    Parameters:
    bounding_box (dict): The bounding box for the region of interest.
    start_time (float): The start time of the analysis in seconds.
    end_time (float): The end time of the analysis in seconds.
    input_video (str): Path to the input video file.
    task_name (str): The name of the task being analyzed.

    Returns:
    dict: The analysis output including display landmarks and normalization factor.
    """
    # Open the video file
    video = cv2.VideoCapture(input_video)

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    start_frame_idx = math.floor(fps * start_time)
    end_frame_idx = math.floor(fps * end_time)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    current_frame_idx = start_frame_idx

    essential_landmarks = []

    # Get the appropriate detector for the task
    detector, detector_update = get_detector(task_name)

    while current_frame_idx < end_frame_idx:
        status, current_frame = video.read()

        if status is False:
            break

        if detector_update:
            detector, _ = get_detector(task_name)

        landmarks = get_essential_landmarks(current_frame, current_frame_idx, task_name, bounding_box, detector)

        # If frame doesn't have essential landmarks, skip
        if not landmarks:
            essential_landmarks.append([])
            current_frame_idx += 1
            continue

        essential_landmarks.append(landmarks)
        current_frame_idx += 1

    # Get display landmarks and normalization factor
    display_landmarks = get_display_landmarks(essential_landmarks, task_name)
    normalization_factor = get_normalisation_factor(essential_landmarks, task_name)

    # Perform further analysis and return the output
    return get_analysis_output(task_name, display_landmarks, normalization_factor, fps, start_time, end_time)

def get_analysis_output(task_name, display_landmarks, normalization_factor, fps, start_time, end_time):
    """
    Get the analysis output for the specified task.

    Parameters:
    task_name (str): The name of the task being analyzed.
    display_landmarks (list): The display landmarks for the task.
    normalization_factor (float): The normalization factor for the task.
    fps (float): The frames per second of the video.
    start_time (float): The start time of the analysis in seconds.
    end_time (float): The end time of the analysis in seconds.

    Returns:
    dict: The analysis output including display landmarks and normalization factor.
    """
    # Get the signal for the task
    task_signal = get_signal(display_landmarks, task_name)
    signal_of_interest = np.array(task_signal) / normalization_factor
    signal_of_interest = filter_signal(signal_of_interest, cut_off_frequency=7.5)

    # Resample the signal to match the time vector
    duration = end_time - start_time
    time_vector = np.linspace(0, duration, int(duration * fps))
    up_sample_signal = signal.resample(signal_of_interest, len(time_vector))

    # Get the final output including additional analysis metrics
    output = get_output(up_sample_signal, duration, start_time)
    output['landMarks'] = display_landmarks
    output['normalization_factor'] = normalization_factor

    return output
