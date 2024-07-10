import numpy as np
import mediapipe as mp
import cv2
import math
import json
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import scipy.signal as signal
import scipy.interpolate as interpolate
import torch
from super_gradients.training import models
from app.analysis.finderPeaksSignal import peakFinder
from app.analysis.analysis import analysis, get_analysis_output

def filterSignal(rawSignal, fs=25, cutOffFrequency=5):
    # Apply a 2nd order Butterworth low-pass filter to the input signal
    # rawSignal: The input signal to be filtered
    # fs: Sampling frequency of the signal (default is 25 Hz)
    # cutOffFrequency: Cut-off frequency for the low-pass filter (default is 5 Hz)
    b, a = signal.butter(2, cutOffFrequency, fs=fs, btype='low', analog=False)
    return signal.filtfilt(b, a, rawSignal)

def run_time():
    # Define the path to the video file
    video_path = '/Users/amergu/Personal/webapps/ml-sample-app/backend/app/uploads/91D4C78179EA446.mp4'
    # Perform leg raise analysis on a specific segment of the video
    # Parameters:
    # - start_time: Start time of the video segment in seconds
    # - end_time: End time of the video segment in seconds
    # - input_video: Path to the input video file
    # - is_left_leg: Boolean indicating if the analysis is for the left leg
    leg_raise_analysis(60, {'x': 263, 'y': 371, 'width': 528, 'height': 1465}, start_time=213.844, end_time=224.937,
                       input_video=video_path, is_left_leg=True)

def json_serialize(obj):
    # Serialize objects to JSON format
    # Special handling for NumPy arrays: convert them to lists
    # For other object types, convert to string representation
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return str(obj)

def leg_raise_analysis(fps, bounding_box, start_time, end_time, input_video, is_left_leg):
    # Define running mode for vision task
    VisionRunningMode = mp.tasks.vision.RunningMode
    # Set base options for the model
    base_options = python.BaseOptions(model_asset_path='app/pose_landmarker_heavy.task')
    # Set options for the pose landmarker
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=VisionRunningMode.VIDEO)

    # Create pose landmarker detector
    detector = vision.PoseLandmarker.create_from_options(options)
    # Open the video file
    video = cv2.VideoCapture(input_video)

    # Calculate the start and end frames
    start_frame = round(fps * start_time)
    end_frame = round(fps * end_time)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frameCounter = start_frame

    # Initialize lists to store landmarks and signals
    knee_landmarks = []
    nose_landmarks = []
    landmarks_signal = []

    # Set knee landmark position based on leg side
    knee_landmark_pos = 26
    nose_landmark_pos = 0

    normalization_factor = 1

    if is_left_leg:
        knee_landmark_pos = 25

    # Process video frames
    while frameCounter < end_frame:
        status, frame = video.read()
        if not status:
            break

        # Convert frame color from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Crop frame based on bounding box info
        x1 = bounding_box['x']
        y1 = bounding_box['y']
        x2 = x1 + bounding_box['width']
        y2 = y1 + bounding_box['height']
        Imagedata = frame[y1:y2, x1:x2, :].astype(np.uint8)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=Imagedata)
        # Detect landmarks in the frame
        detection_result = detector.detect_for_video(image, frameCounter)
        frameCounter += 1

        landmarks = detection_result.pose_landmarks[0]

        # Calculate normalization factor using shoulder and torso midpoints
        if normalization_factor == 1:
            shoulder_mid = [((landmarks[11].x + landmarks[12].x) / 2) * (x2 - x1),
                            ((landmarks[11].y + landmarks[12].y) / 2) * (y2 - y1)]
            torso_mid = [((landmarks[23].x + landmarks[24].x) / 2) * (x2 - x1),
                         ((landmarks[23].y + landmarks[24].y) / 2) * (y2 - y1)]
            normalization_factor = math.dist(shoulder_mid, torso_mid)

        # Calculate distances between knee and nose landmarks
        p = [landmarks[knee_landmark_pos].x * (x2 - x1), landmarks[knee_landmark_pos].y * (y2 - y1)]
        q = [landmarks[nose_landmark_pos].x * (x2 - x1), landmarks[nose_landmark_pos].y * (y2 - y1)]
        landmarks_signal.append([0, (math.dist(p, q) / normalization_factor)])
        knee_landmarks.append(p)
        nose_landmarks.append(q)

    # Convert the landmarks signal to a numpy array and filter it
    signalOfInterest = np.array(landmarks_signal)[:, 1]
    signalOfInterest = filterSignal(signalOfInterest, cutOffFrequency=7.5)

    # Resample the signal to the desired frequency
    currentFs = 1 / fps
    desiredFs = 1 / 60
    duration = end_time - start_time

    timeVector = np.linspace(0, duration, int(duration / currentFs))
    newTimeVector = np.linspace(0, duration, int(duration / desiredFs))
    upsampleSignal = signal.resample(signalOfInterest, len(newTimeVector))

    # Find peaks in the signal using a custom algorithm
    distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity = peakFinder(upsampleSignal, fs=60,
                                                                                         minDistance=3,
                                                                                         cutOffFrequency=7.5, prct=0.05)
    
    # Prepare data for plotting and analysis
    line_time = []
    sizeOfDist = len(distance)
    for index, item in enumerate(distance):
        line_time.append((index / sizeOfDist) * duration + start_time)

    line_peaks = []
    line_peaks_time = []
    line_valleys_start = []
    line_valleys_start_time = []
    line_valleys_end = []
    line_valleys_end_time = []

    line_valleys = []
    line_valleys_time = []

    for index, item in enumerate(peaks):
        line_peaks.append(distance[item['peakIndex']])
        line_peaks_time.append((item['peakIndex'] / sizeOfDist) * duration + start_time)

        line_valleys_start.append(distance[item['openingValleyIndex']])
        line_valleys_start_time.append((item['openingValleyIndex'] / sizeOfDist) * duration + start_time)

        line_valleys_end.append(distance[item['closingValleyIndex']])
        line_valleys_end_time.append((item['closingValleyIndex'] / sizeOfDist) * duration + start_time)

        line_valleys.append(distance[item['openingValleyIndex']])
        line_valleys_time.append((item['openingValleyIndex'] / sizeOfDist) * duration + start_time)

    # Calculate various metrics related to amplitude, velocity, and timing
    amplitude = []
    peakTime = []
    rmsVelocity = []
    maxOpeningSpeed = []
    maxClosingSpeed = []
    averageOpeningSpeed = []
    averageClosingSpeed = []

    for idx, peak in enumerate(peaks):
        x1 = peak['openingValleyIndex']
        y1 = distance[peak['openingValleyIndex']]

        x2 = peak['closingValleyIndex']
        y2 = distance[peak['closingValleyIndex']]

        x = peak['peakIndex']
        y = distance[peak['peakIndex']]

        f = interpolate.interp1d(np.array([x1, x2]), np.array([y1, y2]))

        amplitude.append(y - f(x))

        rmsVelocity.append(np.sqrt(np.mean(velocity[peak['openingValleyIndex']:peak['closingValleyIndex']] ** 2)))

        averageOpeningSpeed.append((y - f(x)) / ((peak['peakIndex'] - peak['openingValleyIndex']) * (1 / 60)))
        averageClosingSpeed.append((y - f(x)) / ((peak['closingValleyIndex'] - peak['peakIndex']) * (1 / 60)))

        peakTime.append(peak['peakIndex'] * (1 / 60))

    meanAmplitude = np.mean(amplitude)
    stdAmplitude = np.std(amplitude)

    meanRMSVelocity = np.mean(rmsVelocity)
    stdRMSVelocity = np.std(rmsVelocity)
    meanAverageOpeningSpeed = np.mean(averageOpeningSpeed)
    stdAverageOpeningSpeed = np.std(averageOpeningSpeed)
    meanAverageClosingSpeed = np.mean(averageClosingSpeed)
    stdAverageClosingSpeed = np.std(averageClosingSpeed)

    meanCycleDuration = np.mean(np.diff(peakTime))
    stdCycleDuration = np.std(np.diff(peakTime))
    rangeCycleDuration = np.max(np.diff(peakTime)) - np.min(np.diff(peakTime))
    rate = len(peaks) / (peaks[-1]['closingValleyIndex'] - peaks[0]['openingValleyIndex']) / (1 / 60)

    earlyPeaks = peaks[:len(peaks) // 3]
    latePeaks = peaks[-len(peaks) // 3:]
    amplitudeDecay = np.mean(distance[:len(peaks) // 3]) / np.mean(distance[-len(peaks) // 3:])
    velocityDecay = np.sqrt(
        np.mean(velocity[earlyPeaks[0]['openingValleyIndex']:earlyPeaks[-1]['closingValleyIndex']] ** 2)) / np.sqrt(
        np.mean(velocity[latePeaks[0]['openingValleyIndex']:latePeaks[-1]['closingValleyIndex']] ** 2))
    rateDecay = (len(earlyPeaks) / (
            (earlyPeaks[-1]['closingValleyIndex'] - earlyPeaks[0]['openingValleyIndex']) / (1 / 60))) / (
                        len(latePeaks) / (
                        (latePeaks[-1]['closingValleyIndex'] - latePeaks[0]['openingValleyIndex']) / (1 / 60)))

    cvAmplitude = stdAmplitude / meanAmplitude
    cvCycleDuration = stdCycleDuration / meanCycleDuration
    cvRMSVelocity = stdRMSVelocity / meanRMSVelocity
    cvAverageOpeningSpeed = stdAverageOpeningSpeed / meanAverageOpeningSpeed
    cvAverageClosingSpeed = stdAverageClosingSpeed / meanAverageClosingSpeed

    # Prepare final JSON object with all the results
    jsonFinal = {
        "linePlot": {
            "data": distance,
            "time": line_time
        },
        "peaks": {
            "data": line_peaks,
            "time": line_peaks_time
        },
        "valleys": {
            "data": line_valleys,
            "time": line_valleys_time
        },
        "valleys_start": {
            "data": line_valleys_start,
            "time": line_valleys_start_time
        },
        "valleys_end": {
            "data": line_valleys_end,
            "time": line_valleys_end_time
        },
        "radar": {
            "A": [2.5, 3.8, 5.9, 4.2, 8.6, 4.9, 6.9, 9, 8.4, 7.3],
            "B": [3.8, 7.1, 7.4, 5.9, 4.2, 3.1, 6.5, 6.4, 3, 7],
            "labels": ["Mean Frequency", "Mean cycle amplitude", "CV cycle amplitude", "Mean cycle rms velocity",
                       "CV cycle rms velocity",
                       "Mean cycle duration", "CV cycle duration", "Range cycle duration", "Amplitude decay",
                       "Velocity decay"],
            "velocity": velocity
        },
        "radarTable": {
            "MeanAmplitude": meanAmplitude,
            "StdAmplitude": stdAmplitude,
            "MeanRMSVelocity": meanRMSVelocity,
            "StdRMSVelocity": stdRMSVelocity,
            "meanAverageOpeningSpeed": meanAverageOpeningSpeed,
            "stdAverageOpeningSpeed": stdAverageOpeningSpeed,
            "meanAverageClosingSpeed": meanAverageClosingSpeed,
            "stdAverageClosingSpeed": stdAverageClosingSpeed,
            "meanCycleDuration": meanCycleDuration,
            "stdCycleDuration": stdCycleDuration,
            "rangeCycleDuration": rangeCycleDuration,
            "rate": rate,
            "amplitudeDecay": amplitudeDecay,
            "velocityDecay": velocityDecay,
            "rateDecay": rateDecay,
            "cvAmplitude": cvAmplitude,
            "cvCycleDuration": cvCycleDuration,
            "cvRMSVelocity": cvRMSVelocity,
            "cvAverageOpeningSpeed": cvAverageOpeningSpeed,
            "cvAverageClosingSpeed": cvAverageClosingSpeed
        },
        "landMarks": knee_landmarks,
        "normalization_landmarks": nose_landmarks,
        "normalization_factor": normalization_factor
    }

    # Serialize JSON object to file
    json_object = json.dumps(jsonFinal, default=json_serialize)
    with open("sample3.json", "w") as outfile:
        outfile.write(json_object)

    return jsonFinal

def toe_tap_analysis(fps, bounding_box, start_time, end_time, input_video, is_left_leg):
    # Define running mode for vision task
    VisionRunningMode = mp.tasks.vision.RunningMode
    # Set base options for the model
    base_options = python.BaseOptions(model_asset_path='app/pose_landmarker_heavy.task')
    # Set options for the pose landmarker
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=VisionRunningMode.VIDEO)
    
    # Create pose landmarker detector
    detector = vision.PoseLandmarker.create_from_options(options)
    # Open the video file
    video = cv2.VideoCapture(input_video)

    # Calculate the start and end frames
    start_frame = round(fps * start_time)
    end_frame = round(fps * end_time)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frameCounter = start_frame

    # Initialize lists to store landmarks and signals
    knee_landmarks = []
    toe_landmarks = []
    landmarks_signal = []

    # Set knee and toe landmark positions based on leg side
    knee_landmark_pos = 26
    toe_landmark_pos = 32

    normalization_factor = 1

    if is_left_leg:
        knee_landmark_pos = 25
        toe_landmark_pos = 31

    # Process video frames
    while frameCounter < end_frame:
        status, frame = video.read()
        if not status:
            break

        # Convert frame color from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Crop frame based on bounding box info
        x1 = bounding_box['x']
        y1 = bounding_box['y']
        x2 = x1 + bounding_box['width']
        y2 = y1 + bounding_box['height']
        Imagedata = frame[y1:y2, x1:x2, :].astype(np.uint8)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=Imagedata)
        # Detect landmarks in the frame
        detection_result = detector.detect_for_video(image, frameCounter)
        frameCounter += 1

        landmarks = detection_result.pose_landmarks[0]

        # Calculate normalization factor using shoulder and torso midpoints
        if normalization_factor == 1:
            shoulder_mid = [((landmarks[11].x + landmarks[12].x) / 2) * (x2 - x1),
                            ((landmarks[11].y + landmarks[12].y) / 2) * (y2 - y1)]
            torso_mid = [((landmarks[23].x + landmarks[24].x) / 2) * (x2 - x1),
                                        ((landmarks[23].y + landmarks[24].y) / 2) * (y2 - y1)]
            normalization_factor = math.dist(shoulder_mid, torso_mid)

        # Calculate distances between toe and knee landmarks
        p = [landmarks[toe_landmark_pos].x * (x2 - x1), landmarks[toe_landmark_pos].y * (y2 - y1)]
        q = [landmarks[knee_landmark_pos].x * (x2 - x1), landmarks[knee_landmark_pos].y * (y2 - y1)]
        landmarks_signal.append([0,(math.dist(p, q)/normalization_factor)])
        toe_landmarks.append(p)
        knee_landmarks.append(q)

    # Convert the landmarks signal to a numpy array and filter it
    signalOfInterest = np.array(landmarks_signal)[:, 1]
    signalOfInterest = filterSignal(signalOfInterest, cutOffFrequency=7.5)

    # Resample the signal to the desired frequency
    currentFs = 1 / fps
    desiredFs = 1 / 60
    duration = end_time - start_time

    timeVector = np.linspace(0, duration, int(duration / currentFs))
    newTimeVector = np.linspace(0, duration, int(duration / desiredFs))
    upsampleSignal = signal.resample(signalOfInterest, len(newTimeVector))

    # Find peaks in the signal using a custom algorithm
    distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity = peakFinder(upsampleSignal, fs=60,
                                                                                         minDistance=3,
                                                                                         cutOffFrequency=7.5, prct=0.05)

    # Prepare data for plotting and analysis
    line_time = []
    sizeOfDist = len(distance)
    for index, item in enumerate(distance):
        line_time.append((index / sizeOfDist) * duration + start_time)

    line_peaks = []
    line_peaks_time = []
    line_valleys_start = []
    line_valleys_start_time = []
    line_valleys_end = []
    line_valleys_end_time = []

    line_valleys = []
    line_valleys_time = []

    for index, item in enumerate(peaks):
        line_peaks.append(distance[item['peakIndex']])
        line_peaks_time.append((item['peakIndex'] / sizeOfDist) * duration + start_time)

        line_valleys_start.append(distance[item['openingValleyIndex']])
        line_valleys_start_time.append((item['openingValleyIndex'] / sizeOfDist) * duration + start_time)

        line_valleys_end.append(distance[item['closingValleyIndex']])
        line_valleys_end_time.append((item['closingValleyIndex'] / sizeOfDist) * duration + start_time)

        line_valleys.append(distance[item['openingValleyIndex']])
        line_valleys_time.append((item['openingValleyIndex'] / sizeOfDist) * duration + start_time)

    # Calculate various metrics related to amplitude, velocity, and timing
    amplitude = []
    peakTime = []
    rmsVelocity = []
    maxOpeningSpeed = []
    maxClosingSpeed = []
    averageOpeningSpeed = []
    averageClosingSpeed = []

    for idx, peak in enumerate(peaks):
        x1 = peak['openingValleyIndex']
        y1 = distance[peak['openingValleyIndex']]

        x2 = peak['closingValleyIndex']
        y2 = distance[peak['closingValleyIndex']]

        x = peak['peakIndex']
        y = distance[peak['peakIndex']]

        f = interpolate.interp1d(np.array([x1, x2]), np.array([y1, y2]))

        amplitude.append(y - f(x))

        rmsVelocity.append(np.sqrt(np.mean(velocity[peak['openingValleyIndex']:peak['closingValleyIndex']] ** 2)))

        averageOpeningSpeed.append((y - f(x)) / ((peak['peakIndex'] - peak['openingValleyIndex']) * (1 / 60)))
        averageClosingSpeed.append((y - f(x)) / ((peak['closingValleyIndex'] - peak['peakIndex']) * (1 / 60)))

        peakTime.append(peak['peakIndex'] * (1 / 60))

    meanAmplitude = np.mean(amplitude)
    stdAmplitude = np.std(amplitude)

    meanRMSVelocity = np.mean(rmsVelocity)
    stdRMSVelocity = np.std(rmsVelocity)
    meanAverageOpeningSpeed = np.mean(averageOpeningSpeed)
    stdAverageOpeningSpeed = np.std(averageOpeningSpeed)
    meanAverageClosingSpeed = np.mean(averageClosingSpeed)
    stdAverageClosingSpeed = np.std(averageClosingSpeed)

    meanCycleDuration = np.mean(np.diff(peakTime))
    stdCycleDuration = np.std(np.diff(peakTime))
    rangeCycleDuration = np.max(np.diff(peakTime)) - np.min(np.diff(peakTime))
    rate = len(peaks) / (peaks[-1]['closingValleyIndex'] - peaks[0]['openingValleyIndex']) / (1 / 60)

    earlyPeaks = peaks[:len(peaks) // 3]
    latePeaks = peaks[-len(peaks) // 3:]
    amplitudeDecay = np.mean(distance[:len(peaks) // 3]) / np.mean(distance[-len(peaks) // 3:])
    velocityDecay = np.sqrt(
        np.mean(velocity[earlyPeaks[0]['openingValleyIndex']:earlyPeaks[-1]['closingValleyIndex']] ** 2)) / np.sqrt(
        np.mean(velocity[latePeaks[0]['openingValleyIndex']:latePeaks[-1]['closingValleyIndex']] ** 2))
    rateDecay = (len(earlyPeaks) / (
            (earlyPeaks[-1]['closingValleyIndex'] - earlyPeaks[0]['openingValleyIndex']) / (1 / 60))) / (
                        len(latePeaks) / (
                        (latePeaks[-1]['closingValleyIndex'] - latePeaks[0]['openingValleyIndex']) / (1 / 60)))

    cvAmplitude = stdAmplitude / meanAmplitude
    cvCycleDuration = stdCycleDuration / meanCycleDuration
    cvRMSVelocity = stdRMSVelocity / meanRMSVelocity
    cvAverageOpeningSpeed = stdAverageOpeningSpeed / meanAverageOpeningSpeed
    cvAverageClosingSpeed = stdAverageClosingSpeed / meanAverageClosingSpeed

    # Prepare final JSON object with all the results
    jsonFinal = {
        "linePlot": {
            "data": distance,
            "time": line_time
        },
        "peaks": {
            "data": line_peaks,
            "time": line_peaks_time
        },
        "valleys": {
            "data": line_valleys,
            "time": line_valleys_time
        },
        "valleys_start": {
            "data": line_valleys_start,
            "time": line_valleys_start_time
        },
        "valleys_end": {
            "data": line_valleys_end,
            "time": line_valleys_end_time
        },
        "radar": {
            "A": [2.5, 3.8, 5.9, 4.2, 8.6, 4.9, 6.9, 9, 8.4, 7.3],
            "B": [3.8, 7.1, 7.4, 5.9, 4.2, 3.1, 6.5, 6.4, 3, 7],
            "labels": ["Mean Frequency", "Mean cycle amplitude", "CV cycle amplitude", "Mean cycle rms velocity",
                       "CV cycle rms velocity",
                       "Mean cycle duration", "CV cycle duration", "Range cycle duration", "Amplitude decay",
                       "Velocity decay"],
            "velocity": velocity
        },
        "radarTable": {
            "MeanAmplitude": meanAmplitude,
            "StdAmplitude": stdAmplitude,
            "MeanRMSVelocity": meanRMSVelocity,
            "StdRMSVelocity": stdRMSVelocity,
            "meanAverageOpeningSpeed": meanAverageOpeningSpeed,
            "stdAverageOpeningSpeed": stdAverageOpeningSpeed,
            "meanAverageClosingSpeed": meanAverageClosingSpeed,
            "stdAverageClosingSpeed": stdAverageClosingSpeed,
            "meanCycleDuration": meanCycleDuration,
            "stdCycleDuration": stdCycleDuration,
            "rangeCycleDuration": rangeCycleDuration,
            "rate": rate,
            "amplitudeDecay": amplitudeDecay,
            "velocityDecay": velocityDecay,
            "rateDecay": rateDecay,
            "cvAmplitude": cvAmplitude,
            "cvCycleDuration": cvCycleDuration,
            "cvRMSVelocity": cvRMSVelocity,
            "cvAverageOpeningSpeed": cvAverageOpeningSpeed,
            "cvAverageClosingSpeed": cvAverageClosingSpeed
        },
        "landMarks": toe_landmarks,
        "normalization_landmarks": knee_landmarks,
        "normalization_factor": normalization_factor
    }

    # Serialize JSON object to file
    json_object = json.dumps(jsonFinal, default=json_serialize)
    with open("toe_tap.json", "w") as outfile:
        outfile.write(json_object)

    return jsonFinal

def updateLandMarks(inputJson):
    # Update landmarks based on input JSON
    task_name = inputJson['task_name']
    display_landmarks = inputJson['landmarks']
    fps = inputJson['fps']
    start_time =  inputJson['start_time'] 
    end_time =  inputJson['end_time']
    normalization_factor = 1
    if 'normalization_factor' in inputJson:
        normalization_factor = inputJson['normalization_factor']

    # Get analysis output
    return get_analysis_output(task_name, display_landmarks, normalization_factor, fps, start_time, end_time)

def updatePeaksAndValleys(inputJson):
    # Update peaks and valleys based on input JSON
    peaksData = inputJson['peaks_Data']
    peaksTime = inputJson['peaks_Time']
    valleysStartData =  inputJson['valleys_StartData'] 
    valleysStartTime =  inputJson['valleys_StartTime']
    valleysEndData =  inputJson['valleys_EndData']
    valleysEndTime =  inputJson['valleys_EndTime']
    velocity =  inputJson['_velocity']

    # Sort valleysStartTime and get the permutation indices
    sorted_indices = sorted(range(len(valleysStartTime)), key=lambda k: valleysStartTime[k])

    # Rearrange valleysStartTime
    valleysStartTime_sorted = sorted(valleysStartTime)

    # Rearrange valleysStartData based on sorted_indices
    valleysStartData_sorted = [valleysStartData[i] for i in sorted_indices]

    # Sort valleysEndTime and get the permutation indices
    sorted_indices_end = sorted(range(len(valleysEndTime)), key=lambda k: valleysEndTime[k])

    # Rearrange valleysEndTime
    valleysEndTime_sorted = sorted(valleysEndTime)

    # Rearrange valleysEndData based on sorted_indices_end
    valleysEndData_sorted = [valleysEndData[i] for i in sorted_indices_end]

    # Sort peaksTime and get the permutation indices
    sorted_indices_peaks = sorted(range(len(peaksTime)), key=lambda k: peaksTime[k])

    # Rearrange peaksTime
    peaksTime_sorted = sorted(peaksTime)

    # Rearrange peaksData based on sorted_indices_peaks
    peaksData_sorted = [peaksData[i] for i in sorted_indices_peaks]

    peaksTime = peaksTime_sorted
    peaksData = peaksData_sorted
    valleysEndTime = valleysEndTime_sorted
    valleysEndData = valleysEndData_sorted
    valleysStartTime = valleysStartTime_sorted
    valleysStartData = valleysStartData_sorted

    amplitude = []
    peakTime = []
    rmsVelocity = []
    averageOpeningSpeed = []
    averageClosingSpeed = []

    for idx, item in enumerate(peaksData):
        # Calculate height measures
        x1 = valleysStartTime[idx]
        y1 = valleysStartData[idx]

        x2 = valleysEndTime[idx]
        y2 = valleysEndData[idx]

        x = peaksTime[idx]
        y = peaksData[idx]

        f = interpolate.interp1d(np.array([x1, x2]), np.array([y1, y2]))

        amplitude.append(y - f(x))

        # Calculate opening velocity
        # rmsVelocity.append(np.sqrt(np.mean(velocity[valleysStartTime[idx]:valleysEndTime[idx]] ** 2)))

        averageOpeningSpeed.append((y - f(x)) / ((peaksTime[idx] - valleysStartTime[idx]) * (1 / 60)))
        averageClosingSpeed.append((y - f(x)) / ((valleysEndTime[idx] - peaksTime[idx]) * (1 / 60)))

        peakTime.append(peaksTime[idx] * (1 / 60))

    meanAmplitude = np.mean(amplitude)
    stdAmplitude = np.std(amplitude)

    # meanRMSVelocity = np.mean(rmsVelocity)
    # stdRMSVelocity = np.std(rmsVelocity)
    meanAverageOpeningSpeed = np.mean(averageOpeningSpeed)
    stdAverageOpeningSpeed = np.std(averageOpeningSpeed)
    meanAverageClosingSpeed = np.mean(averageClosingSpeed)
    stdAverageClosingSpeed = np.std(averageClosingSpeed)

    meanCycleDuration = np.mean(np.diff(peakTime))
    stdCycleDuration = np.std(np.diff(peakTime))
    rangeCycleDuration = np.max(np.diff(peakTime)) - np.min(np.diff(peakTime))
    rate = len(peaksTime) / (valleysEndTime[0] - valleysStartTime[0]) / (1 / 60)

    cvAmplitude = stdAmplitude / meanAmplitude
    cvCycleDuration = stdCycleDuration / meanCycleDuration
    # cvRMSVelocity = stdRMSVelocity / meanRMSVelocity
    cvAverageOpeningSpeed = stdAverageOpeningSpeed / meanAverageOpeningSpeed
    cvAverageClosingSpeed = stdAverageClosingSpeed / meanAverageClosingSpeed

    # Prepare radar table with all the results
    radarTable = {
        "MeanAmplitude": meanAmplitude,
        "StdAmplitude": stdAmplitude,
        "meanAverageOpeningSpeed": meanAverageOpeningSpeed,
        "stdAverageOpeningSpeed": stdAverageOpeningSpeed,
        "meanAverageClosingSpeed": meanAverageClosingSpeed,
        "stdAverageClosingSpeed": stdAverageClosingSpeed,
        "meanCycleDuration": meanCycleDuration,
        "stdCycleDuration": stdCycleDuration,
        "rangeCycleDuration": rangeCycleDuration,
        "rate": rate,
        "cvAmplitude": cvAmplitude,
        "cvCycleDuration": cvCycleDuration,
        "cvAverageOpeningSpeed": cvAverageOpeningSpeed,
        "cvAverageClosingSpeed": cvAverageClosingSpeed
    }
    
    return radarTable

def leg_raise_yolo(fps, bounding_box, start_time, end_time, input_video, is_left_leg):

    # Load YOLO model
    model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    confidence = 0.7

    # Open the video file
    video = cv2.VideoCapture(input_video)

    # Calculate the start and end frames
    start_frame = round(fps * start_time)
    end_frame = round(fps * end_time)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frameCounter = start_frame

    # Initialize lists to store landmarks and signals
    knee_landmarks = []
    shoulder_landmarks = []
    landmarks_signal = []

    knee_landmark_pos = 14

    if is_left_leg:
        knee_landmark_pos = 13

    torso = []
    # Process video frames
    while frameCounter < end_frame:
        status, frame = video.read()
        if not status:
            break

        # Crop frame based on bounding box info
        x1 = bounding_box['x']
        y1 = bounding_box['y']
        x2 = x1 + bounding_box['width']
        y2 = y1 + bounding_box['height']

        # Extracting the ROI based on the bounding box
        roi = frame[y1:y2, x1:x2]

        # Convert the ROI to RGB since many models expect input in this format
        results = model.predict(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), conf=confidence)
        landmarks = results.prediction.poses[0]
        frameCounter += 1

        midpointshoulders = (np.array(landmarks[5, :2]) + np.array(landmarks[6, :2])) / 2
        midpointhips = (np.array(landmarks[11, :2]) + np.array(landmarks[12, :2])) / 2

        landmarks_signal.append(np.linalg.norm(np.array(landmarks[knee_landmark_pos, :2]) - midpointshoulders))
        shoulder_landmarks.append(midpointshoulders)
        knee_landmarks.append(landmarks[knee_landmark_pos])

        torso.append(np.abs(midpointshoulders[1] - midpointhips[1]))

    normalization_factor = np.mean(torso)

    signalOfInterest = np.array(landmarks_signal)/normalization_factor
    signalOfInterest = filterSignal(signalOfInterest, cutOffFrequency=8)

    # Resample the signal to the desired frequency
    currentFs = 1 / fps
    desiredFs = 1 / 60
    duration = end_time - start_time
    print(duration)
    print(knee_landmarks)
    print(signalOfInterest)

    timeVector = np.linspace(0, duration, int(duration / currentFs))
    newTimeVector = np.linspace(0, duration, int(duration / desiredFs))
    upsampleSignal = signal.resample(signalOfInterest, len(newTimeVector))

    # Find peaks in the signal using a custom algorithm
    distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity = peakFinder(upsampleSignal, fs=60,
                                                                                         minDistance=3,
                                                                                         cutOffFrequency=7.5, prct=0.05)

    # Prepare data for plotting and analysis
    line_time = []
    sizeOfDist = len(distance)
    for index, item in enumerate(distance):
        line_time.append((index / sizeOfDist) * duration + start_time)

    line_peaks = []
    line_peaks_time = []
    line_valleys_start = []
    line_valleys_start_time = []
    line_valleys_end = []
    line_valleys_end_time = []

    line_valleys = []
    line_valleys_time = []

    for index, item in enumerate(peaks):
        line_peaks.append(distance[item['peakIndex']])
        line_peaks_time.append((item['peakIndex'] / sizeOfDist) * duration + start_time)

        line_valleys_start.append(distance[item['openingValleyIndex']])
        line_valleys_start_time.append((item['openingValleyIndex'] / sizeOfDist) * duration + start_time)

        line_valleys_end.append(distance[item['closingValleyIndex']])
        line_valleys_end_time.append((item['closingValleyIndex'] / sizeOfDist) * duration + start_time)

        line_valleys.append(distance[item['openingValleyIndex']])
        line_valleys_time.append((item['openingValleyIndex'] / sizeOfDist) * duration + start_time)

    # Calculate various metrics related to amplitude, velocity, and timing
    amplitude = []
    peakTime = []
    rmsVelocity = []
    maxOpeningSpeed = []
    maxClosingSpeed = []
    averageOpeningSpeed = []
    averageClosingSpeed = []

    for idx, peak in enumerate(peaks):
        x1 = peak['openingValleyIndex']
        y1 = distance[peak['openingValleyIndex']]

        x2 = peak['closingValleyIndex']
        y2 = distance[peak['closingValleyIndex']]

        x = peak['peakIndex']
        y = distance[peak['peakIndex']]

        f = interpolate.interp1d(np.array([x1, x2]), np.array([y1, y2]))

        amplitude.append(y - f(x))

        rmsVelocity.append(np.sqrt(np.mean(velocity[peak['openingValleyIndex']:peak['closingValleyIndex']] ** 2)))

        averageOpeningSpeed.append((y - f(x)) / ((peak['peakIndex'] - peak['openingValleyIndex']) * (1 / 60)))
        averageClosingSpeed.append((y - f(x)) / ((peak['closingValleyIndex'] - peak['peakIndex']) * (1 / 60)))

        peakTime.append(peak['peakIndex'] * (1 / 60))

    meanAmplitude = np.mean(amplitude)
    stdAmplitude = np.std(amplitude)

    meanRMSVelocity = np.mean(rmsVelocity)
    stdRMSVelocity = np.std(rmsVelocity)
    meanAverageOpeningSpeed = np.mean(averageOpeningSpeed)
    stdAverageOpeningSpeed = np.std(averageOpeningSpeed)
    meanAverageClosingSpeed = np.mean(averageClosingSpeed)
    stdAverageClosingSpeed = np.std(averageClosingSpeed)

    meanCycleDuration = np.mean(np.diff(peakTime))
    stdCycleDuration = np.std(np.diff(peakTime))
    rangeCycleDuration = np.max(np.diff(peakTime)) - np.min(np.diff(peakTime))
    rate = len(peaks) / (peaks[-1]['closingValleyIndex'] - peaks[0]['openingValleyIndex']) / (1 / 60)

    earlyPeaks = peaks[:len(peaks) // 3]
    latePeaks = peaks[-len(peaks) // 3:]
    amplitudeDecay = np.mean(distance[:len(peaks) // 3]) / np.mean(distance[-len(peaks) // 3:])
    velocityDecay = np.sqrt(
        np.mean(velocity[earlyPeaks[0]['openingValleyIndex']:earlyPeaks[-1]['closingValleyIndex']] ** 2)) / np.sqrt(
        np.mean(velocity[latePeaks[0]['openingValleyIndex']:latePeaks[-1]['closingValleyIndex']] ** 2))
    rateDecay = (len(earlyPeaks) / (
            (earlyPeaks[-1]['closingValleyIndex'] - earlyPeaks[0]['openingValleyIndex']) / (1 / 60))) / (
                        len(latePeaks) / (
                        (latePeaks[-1]['closingValleyIndex'] - latePeaks[0]['openingValleyIndex']) / (1 / 60)))

    cvAmplitude = stdAmplitude / meanAmplitude
    cvCycleDuration = stdCycleDuration / meanCycleDuration
    cvRMSVelocity = stdRMSVelocity / meanRMSVelocity
    cvAverageOpeningSpeed = stdAverageOpeningSpeed / meanAverageOpeningSpeed
    cvAverageClosingSpeed = stdAverageClosingSpeed / meanAverageClosingSpeed

    # Prepare final JSON object with all the results
    jsonFinal = {
        "linePlot": {
            "data": distance,
            "time": line_time
        },
        "peaks": {
            "data": line_peaks,
            "time": line_peaks_time
        },
        "valleys": {
            "data": line_valleys,
            "time": line_valleys_time
        },
        "valleys_start": {
            "data": line_valleys_start,
            "time": line_valleys_start_time
        },
        "valleys_end": {
            "data": line_valleys_end,
            "time": line_valleys_end_time
        },
        "radar": {
            "A": [2.5, 3.8, 5.9, 4.2, 8.6, 4.9, 6.9, 9, 8.4, 7.3],
            "B": [3.8, 7.1, 7.4, 5.9, 4.2, 3.1, 6.5, 6.4, 3, 7],
            "labels": ["Mean Frequency", "Mean cycle amplitude", "CV cycle amplitude", "Mean cycle rms velocity",
                       "CV cycle rms velocity",
                       "Mean cycle duration", "CV cycle duration", "Range cycle duration", "Amplitude decay",
                       "Velocity decay"],
            "velocity": velocity
        },
        "radarTable": {
            "MeanAmplitude": meanAmplitude,
            "StdAmplitude": stdAmplitude,
            "MeanRMSVelocity": meanRMSVelocity,
            "StdRMSVelocity": stdRMSVelocity,
            "meanAverageOpeningSpeed": meanAverageOpeningSpeed,
            "stdAverageOpeningSpeed": stdAverageOpeningSpeed,
            "meanAverageClosingSpeed": meanAverageClosingSpeed,
            "stdAverageClosingSpeed": stdAverageClosingSpeed,
            "meanCycleDuration": meanCycleDuration,
            "stdCycleDuration": stdCycleDuration,
            "rangeCycleDuration": rangeCycleDuration,
            "rate": rate,
            "amplitudeDecay": amplitudeDecay,
            "velocityDecay": velocityDecay,
            "rateDecay": rateDecay,
            "cvAmplitude": cvAmplitude,
            "cvCycleDuration": cvCycleDuration,
            "cvRMSVelocity": cvRMSVelocity,
            "cvAverageOpeningSpeed": cvAverageOpeningSpeed,
            "cvAverageClosingSpeed": cvAverageClosingSpeed
        },
        "landMarks": knee_landmarks,
        "normalization_landmarks": shoulder_landmarks,
        "normalization_factor": normalization_factor
    }

    # Serialize JSON object to file
    json_object = json.dumps(jsonFinal, default=json_serialize)
    # Writing to JSON file
    file_name = "leg_agility_left" if is_left_leg else "leg_agility_right"

    with open(file_name + ".json", "w") as outfile:
        outfile.write(json_object)

    return jsonFinal

def final_analysis(inputJson, inputVideo):
    # Read bounding box information from input JSON
    data = inputJson
    boundingBox = data['boundingBox']
    fps = data['fps']
    start_time = data['start_time']
    end_time = data['end_time']
    task_name = data['task_name']
    
    # Perform analysis based on the task name
    return analysis(boundingBox, start_time, end_time, inputVideo, task_name)


def final_analysis(inputJson, inputVideo):
    # %%
    # Read boundingbox file
    # f = open('leg_raise_task_data.json')

    # returns JSON object as
    # a dictionary
    # data = json.load(f)
    data = inputJson
    boundingBox = data['boundingBox']
    fps = data['fps']
    start_time = data['start_time']
    end_time = data['end_time']
    task_name = data['task_name']
    
    return analysis(boundingBox, start_time, end_time, inputVideo, task_name)

    def final_analysis(inputJson, inputVideo):
    # Read bounding box information from input JSON
    data = inputJson
    boundingBox = data['boundingBox']
    fps = data['fps']
    start_time = data['start_time']
    end_time = data['end_time']
    task_name = data['task_name']
    
    # Perform analysis based on the task name
    return analysis(boundingBox, start_time, end_time, inputVideo, task_name)

# Paste here 
def final_analysis(inputJson, inputVideo):
    # Read input JSON file
    data = inputJson
    boundingBox = data['boundingBox']  # Get bounding box information
    fps = data['fps']  # Get frames per second from JSON data
    start_time = data['start_time']  # Get start time for analysis
    end_time = data['end_time']  # Get end time for analysis
    task_name = data['task_name']  # Get task name

    # Call the analysis function with extracted parameters
    return analysis(boundingBox, start_time, end_time, inputVideo, task_name)

    # The following commented-out code is an alternative way to handle different tasks,
    # which is currently not in use. The approach used above simplifies this.

    # if task_name == 'Leg agility - Right':
    #     return leg_raise_yolo(fps, boundingBox, start_time, end_time, inputVideo, False)
    # elif task_name == 'Leg agility - Left':
    #     return leg_raise_yolo(fps, boundingBox, start_time, end_time, inputVideo, True)
    # elif task_name == 'Toe tapping - Left':
    #     return toe_tap_analysis(fps, boundingBox, start_time, end_time, inputVideo, True)
    # elif task_name == 'Toe tapping - Right':
    #     return toe_tap_analysis(fps, boundingBox, start_time, end_time, inputVideo, False)
    # elif task_name == 'Finger Tap - Left':
    #     return finger_tap(fps, boundingBox, start_time, end_time, inputVideo, True)
    # elif task_name == 'Finger Tap - Right':
    #     return finger_tap(fps, boundingBox, start_time, end_time, inputVideo, False)
    # elif task_name == 'Hand movement - Left':
    #     return hand_analysis(fps, boundingBox, start_time, end_time, inputVideo, True)
    # elif task_name == 'Hand movement - Right':
    #     return hand_analysis(fps, boundingBox, start_time, end_time, inputVideo, False)

    # The code above simplifies the analysis function by directly calling the analysis method
    # instead of using multiple if-else statements to handle different tasks.

    # Closing file
    # f.close()

    # The commented code below is a more detailed way of handling different tasks.
    # It's kept for reference but not currently used.

    # VisionRunningMode = mp.tasks.vision.RunningMode
    # base_options = python.BaseOptions(model_asset_path='app/pose_landmarker_heavy.task')
    # options = vision.PoseLandmarkerOptions(
    #     base_options=base_options,
    #     output_segmentation_masks=False,
    #     running_mode=VisionRunningMode.VIDEO)
    # detector = vision.PoseLandmarker.create_from_options(options)
    # video = cv2.VideoCapture(inputVideo)
    # video.set(cv2.CAP_PROP_POS_FRAMES, )
    # frameCounter = 0
    #
    # rightKneeLandmarks = []
    #
    # while True:
    #     status, frame = video.read()
    #     if status == False:
    #         break
    #
    #     # detect landmarks
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #
    #     # crop frame based on bounding box info
    #     x1 = boundingBoxes[frameCounter]['data'][0]['x']
    #     y1 = boundingBoxes[frameCounter]['data'][0]['y']
    #     x2 = x1 + boundingBoxes[frameCounter]['data'][0]['width']
    #     y2 = y1 + boundingBoxes[frameCounter]['data'][0]['height']
    #     Imagedata = frame[y1:y2, x1:x2, :].astype(np.uint8)
    #     image = mp.Image(image_format=mp.ImageFormat.SRGB, data=Imagedata)
    #     detection_result = detector.detect_for_video(image, frameCounter)
    #     frameCounter = frameCounter + 1
    #
    #     landmarks = detection_result.pose_landmarks[0]
    #     # these are the coordinates of the landmark that you want to display in the video
    #     rightKneeLandmarks.append([landmarks[26].x * (x2 - x1), landmarks[26].y * (y2 - y1), landmarks[26].z])
    #
    # signalOfInterest = np.array(rightKneeLandmarks)[:, 1]
    # signalOfInterest = filterSignal(signalOfInterest, cutOffFrequency=7.5)
    #
    # # find peaks using custom algorithm
    # currentFs = 1 / int(data['fps'])
    # desiredFs = 1 / 60
    #
    # duration = len(signalOfInterest) * (currentFs)
    # print(duration)
    #
    # timeVector = np.linspace(0, duration, int(duration / currentFs))
    #
    # newTimeVector = np.linspace(0, duration, int(duration / desiredFs))
    # upsampleSignal = signal.resample(signalOfInterest, len(newTimeVector))
    #
    # distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity = peakFinder(upsampleSignal, fs=60,
    #                                                                                      minDistance=3,
    #                                                                                      cutOffFrequency=7.5, prct=0.05)
    # #plot
    # #a plot like this should appear on the page.
    #
    # #green dots -> Peaks
    # #red dots -> Left Valleys
    # #blue dots -> Right Valleys
    # figure, ax = plt.subplots(figsize=(15, 5))
    # ax.plot(range(len(distance)), distance)
    # ax.set_ylim([np.min(distance), np.max(distance)])
    # ax.set_xlim([0, len(distance)])
    # print(duration)
    # print(frameCounter)
    # line_time = []
    # sizeOfDist = len(distance)
    # for index, item in enumerate(distance):
    #     line_time.append((index / sizeOfDist) * duration)
    #
    # line_peaks = []
    # line_peaks_time = []
    # line_valleys_start = []
    # line_valleys_start_time = []
    # line_valleys_end = []
    # line_valleys_end_time = []
    #
    # line_valleys = []
    # line_valleys_time = []
    #
    # for index, item in enumerate(peaks):
    #     ax.plot(item['openingValleyIndex'], distance[item['openingValleyIndex']], 'ro', alpha=0.75)
    #     ax.plot(item['peakIndex'], distance[item['peakIndex']], 'go', alpha=0.75)
    #     ax.plot(item['closingValleyIndex'], distance[item['closingValleyIndex']], 'bo', alpha=0.75)
    #     # line_valleys.append(prevValley+item['openingValleyIndex'])
    #
    #     line_peaks.append(distance[item['peakIndex']])
    #     line_peaks_time.append((item['peakIndex'] / sizeOfDist) * duration)
    #
    #     line_valleys_start.append(distance[item['openingValleyIndex']])
    #     line_valleys_start_time.append((item['openingValleyIndex'] / sizeOfDist) * duration)
    #
    #     line_valleys_end.append(distance[item['closingValleyIndex']])
    #     line_valleys_end_time.append((item['closingValleyIndex'] / sizeOfDist) * duration)
    #
    #     line_valleys.append(distance[item['openingValleyIndex']])
    #     line_valleys_time.append((item['openingValleyIndex'] / sizeOfDist) * duration)
    #
    # amplitude = []
    # peakTime = []
    # rmsVelocity = []
    # maxOpeningSpeed = []
    # maxClosingSpeed = []
    # averageOpeningSpeed = []
    # averageClosingSpeed = []
    #
    # for idx, peak in enumerate(peaks):
    #     # Height measures
    #     x1 = peak['openingValleyIndex']
    #     y1 = distance[peak['openingValleyIndex']]
    #
    #     x2 = peak['closingValleyIndex']
    #     y2 = distance[peak['closingValleyIndex']]
    #
    #     x = peak['peakIndex']
    #     y = distance[peak['peakIndex']]
    #
    #     f = interpolate.interp1d(np.array([x1, x2]), np.array([y1, y2]))
    #
    #     amplitude.append(y - f(x))
    #
    #     # Opening Velocity
    #     rmsVelocity.append(np.sqrt(np.mean(velocity[peak['openingValleyIndex']:peak['closingValleyIndex']] ** 2)))
    #
    #     averageOpeningSpeed.append((y - f(x)) / ((peak['peakIndex'] - peak['openingValleyIndex']) * (1 / 60)))
    #     averageClosingSpeed.append((y - f(x)) / ((peak['closingValleyIndex'] - peak['peakIndex']) * (1 / 60)))
    #
    #     # timming
    #     peakTime.append(peak['peakIndex'] * (1 / 60))
    #
    # meanAmplitude = np.mean(amplitude)
    # stdAmplitude = np.std(amplitude)
    #
    # meanRMSVelocity = np.mean(rmsVelocity)
    # stdRMSVelocity = np.std(rmsVelocity)
    # meanAverageOpeningSpeed = np.mean(averageOpeningSpeed)
    # stdAverageOpeningSpeed = np.std(averageOpeningSpeed)
    # meanAverageClosingSpeed = np.mean(averageClosingSpeed)
    # stdAverageClosingSpeed = np.std(averageClosingSpeed)
    #
    # meanCycleDuration = np.mean(np.diff(peakTime))
    # stdCycleDuration = np.std(np.diff(peakTime))
    # rangeCycleDuration = np.max(np.diff(peakTime)) - np.min(np.diff(peakTime))
    # rate = len(peaks) / (peaks[-1]['closingValleyIndex'] - peaks[0]['openingValleyIndex']) / (1 / 60)
    #
    # earlyPeaks = peaks[:len(peaks) // 3]
    # latePeaks = peaks[-len(peaks) // 3:]
    # amplitudeDecay = np.mean(distance[:len(peaks) // 3]) / np.mean(distance[-len(peaks) // 3:])
    # velocityDecay = np.sqrt(
    #     np.mean(velocity[earlyPeaks[0]['openingValleyIndex']:earlyPeaks[-1]['closingValleyIndex']] ** 2)) / np.sqrt(
    #     np.mean(velocity[latePeaks[0]['openingValleyIndex']:latePeaks[-1]['closingValleyIndex']] ** 2))
    # rateDecay = (len(earlyPeaks) / (
    #             (earlyPeaks[-1]['closingValleyIndex'] - earlyPeaks[0]['openingValleyIndex']) / (1 / 60))) / (
    #                         len(latePeaks) / (
    #                             (latePeaks[-1]['closingValleyIndex'] - latePeaks[0]['openingValleyIndex']) / (1 / 60)))
    #
    # cvAmplitude = stdAmplitude / meanAmplitude
    # cvCycleDuration = stdCycleDuration / meanCycleDuration
    # cvRMSVelocity = stdRMSVelocity / meanRMSVelocity
    # cvAverageOpeningSpeed = stdAverageOpeningSpeed / meanAverageOpeningSpeed
    # cvAverageClosingSpeed = stdAverageClosingSpeed / meanAverageClosingSpeed
    #
    # jsonFinal = {
    #     "linePlot": {
    #         "data": distance,
    #         "time": line_time
    #     },
    #     "peaks": {
    #         "data": line_peaks,
    #         "time": line_peaks_time
    #     },
    #     "valleys": {
    #         "data": line_valleys,
    #         "time": line_valleys_time
    #     },
    #     "valleys_start": {
    #         "data": line_valleys_start,
    #         "time": line_valleys_start_time
    #     },
    #     "valleys_end": {
    #         "data": line_valleys_end,
    #         "time": line_valleys_end_time
    #     },
    #     "radar": {
    #         "A": [2.5, 3.8, 5.9, 4.2, 8.6, 4.9, 6.9, 9, 8.4, 7.3],
    #         "B": [3.8, 7.1, 7.4, 5.9, 4.2, 3.1, 6.5, 6.4, 3, 7],
    #         "labels": ["Mean Frequency", "Mean cycle amplitude", "CV cycle amplitude", "Mean cycle rms velocity",
    #                    "CV cycle rms velocity",
    #                    "Mean cycle duration", "CV cycle duration", "Range cycle duration", "Amplitude decay",
    #                    "Velocity decay"]
    #     },
    #     "radarTable": {
    #         "MeanAmplitude": meanAmplitude,
    #         "StdAmplitude": stdAmplitude,
    #         "MeanRMSVelocity": meanRMSVelocity,
    #         "StdRMSVelocity": stdRMSVelocity,
    #         "meanAverageOpeningSpeed": meanAverageOpeningSpeed,
    #         "stdAverageOpeningSpeed": stdAverageOpeningSpeed,
    #         "meanAverageClosingSpeed": meanAverageClosingSpeed,
    #         "stdAverageClosingSpeed": stdAverageClosingSpeed,
    #         "meanCycleDuration": meanCycleDuration,
    #         "stdCycleDuration": stdCycleDuration,
    #         "rangeCycleDuration": rangeCycleDuration,
    #         "rate": rate,
    #         "amplitudeDecay": amplitudeDecay,
    #         "velocityDecay": velocityDecay,
    #         "rateDecay": rateDecay,
    #         "cvAmplitude": cvAmplitude,
    #         "cvCycleDuration": cvCycleDuration,
    #         "cvRMSVelocity": cvRMSVelocity,
    #         "cvAverageOpeningSpeed": cvAverageOpeningSpeed,
    #         "cvAverageClosingSpeed": cvAverageClosingSpeed
    #     },
    #     "landMarks": rightKneeLandmarks
    #
    # }
    # return jsonFinal

# The commented code below is for plotting and additional calculations,
# which is currently not used but kept for reference.

# for item in peaks :
#     ax.plot(item['openingValleyIndex'], distance[item['openingValleyIndex']],'ro', alpha=0.75)
#     ax.plot(item['peakIndex'], distance[item['peakIndex']],'go', alpha=0.75)
#     ax.plot(item['closingValleyIndex'], distance[item['closingValleyIndex']],'bo', alpha=0.75)
#
# plt.show()
#
# amplitude  = []
# peakTime = []
# rmsVelocity = []
# maxOpeningSpeed = []
# maxClosingSpeed = []
# averageOpeningSpeed = []
# averageClosingSpeed = []
#
# for idx,peak in enumerate(peaks):
#
#     #Height measures
#     x1 = peak['openingValleyIndex']
#     y1 = distance[peak['openingValleyIndex']]
#
#     x2 = peak['closingValleyIndex']
#     y2 = distance[peak['closingValleyIndex']]
#
#     x = peak['peakIndex']
#     y = distance[peak['peakIndex']]
#
#     f = interpolate.interp1d(np.array([x1,x2]),np.array([y1,y2]))
#
#     amplitude.append(y-f(x))
#
#     #Opening Velocity
#     rmsVelocity.append(np.sqrt(np.mean(velocity[peak['openingValleyIndex']:peak['closingValleyIndex']]**2)))
#
#     averageOpeningSpeed.append((y-f(x))/((peak['peakIndex']-peak['openingValleyIndex'])*(1/60)))
#     averageClosingSpeed.append((y-f(x))/((peak['closingValleyIndex']-peak['peakIndex'])*(1/60)))
#
#     #timming
#     peakTime.append(peak['peakIndex']*(1/60))
#
# meanAmplitude = np.mean(amplitude)
# stdAmplitude = np.std(amplitude)
#
# meanRMSVelocity = np.mean(rmsVelocity)
# stdRMSVelocity = np.std(rmsVelocity)
# meanAverageOpeningSpeed = np.mean(averageOpeningSpeed)
# stdAverageOpeningSpeed =  np.std(averageOpeningSpeed)
# meanAverageClosingSpeed = np.mean(averageClosingSpeed)
# stdAverageClosingSpeed =  np.std(averageClosingSpeed)
#
# meanCycleDuration = np.mean(np.diff(peakTime))
# stdCycleDuration = np.std(np.diff(peakTime))
# rangeCycleDuration = np.max(np.diff(peakTime)) - np.min(np.diff(peakTime))
# rate = len(peaks) / (peaks[-1]['closingValleyIndex'] - peaks[0]['openingValleyIndex'])/(1/60)
#
# earlyPeaks = peaks[:len(peaks)//3]
# latePeaks = peaks[-len(peaks)//3:]
# amplitudeDecay = np.mean(distance[:len(peaks)//3])/np.mean(distance[-len(peaks)//3:])
# velocityDecay = np.sqrt(np.mean(velocity[earlyPeaks[0]['openingValleyIndex']:earlyPeaks[-1]['closingValleyIndex']]**2)) / np.sqrt(np.mean(velocity[latePeaks[0]['openingValleyIndex']:latePeaks[-1]['closingValleyIndex']]**2))
# rateDecay = (len(earlyPeaks) / ((earlyPeaks[-1]['closingValleyIndex'] - earlyPeaks[0]['openingValleyIndex'])/(1/60)) ) / ( len(latePeaks) / ((latePeaks[-1]['closingValleyIndex'] - latePeaks[0]['openingValleyIndex'])/(1/60)))
#
# cvAmplitude = stdAmplitude/meanAmplitude
# cvCycleDuration = stdCycleDuration/meanCycleDuration
# cvRMSVelocity = stdRMSVelocity/meanRMSVelocity
# cvAverageOpeningSpeed = stdAverageOpeningSpeed/meanAverageOpeningSpeed
# cvAverageClosingSpeed = stdAverageClosingSpeed/meanAverageClosingSpeed
#
# results = np.array([meanAmplitude,stdAmplitude,\
#                     meanRMSVelocity, stdRMSVelocity, meanAverageOpeningSpeed, stdAverageOpeningSpeed, meanAverageClosingSpeed, stdAverageClosingSpeed,\
#                     meanCycleDuration,stdCycleDuration,rangeCycleDuration,rate,\
#                     amplitudeDecay,velocityDecay,rateDecay,\
#                     cvAmplitude, cvCycleDuration, cvRMSVelocity, cvAverageOpeningSpeed, cvAverageClosingSpeed])
