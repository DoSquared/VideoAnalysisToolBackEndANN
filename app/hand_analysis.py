def finger_tap(fps, bounding_box, start_time, end_time, input_video, is_left_leg):
    # Define VisionRunningMode and base options for the hand landmark detector
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path='app/models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2, running_mode=VisionRunningMode.VIDEO)

    # Create the hand landmark detector
    detector = vision.HandLandmarker.create_from_options(options=options)
    
    # Open the input video file
    video = cv2.VideoCapture(input_video)

    # Calculate the start and end frames based on the given start and end times
    start_frame = round(fps * start_time)
    end_frame = round(fps * end_time)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frameCounter = start_frame

    # Initialize lists to store landmarks and signals
    knee_landmarks = []
    nose_landmarks = []
    landmarks_signal = []

    # Define landmark positions for knee and nose
    knee_landmark_pos = 8
    nose_landmark_pos = 4

    # Set the normalization factor
    normalization_factor = 1

    # Adjust knee landmark position based on whether it's the left leg
    if is_left_leg:
        knee_landmark_pos = 8

    while frameCounter < end_frame:
        status, frame = video.read()
        if not status:
            break

        # Convert frame color to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Crop the frame based on the bounding box information
        x1 = bounding_box['x']
        y1 = bounding_box['y']
        x2 = x1 + bounding_box['width']
        y2 = y1 + bounding_box['height']
        Imagedata = frame[y1:y2, x1:x2, :].astype(np.uint8)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=Imagedata)

        # Detect landmarks in the frame
        detection_result = detector.detect_for_video(image, frameCounter)
        frameCounter += 1

        # Determine the index of the hand based on handedness and leg side
        if is_left_leg:
            index = 0 if detection_result.handedness[0][0].category_name == 'Left' else 1
        else:
            index = 0 if detection_result.handedness[0][0].category_name == 'Right' else 1

        landmarks = detection_result.hand_landmarks[index]

        # Calculate the positions of the knee and nose landmarks
        p = [landmarks[knee_landmark_pos].x * (x2 - x1), landmarks[knee_landmark_pos].y * (y2 - y1)]
        q = [landmarks[nose_landmark_pos].x * (x2 - x1), landmarks[nose_landmark_pos].y * (y2 - y1)]
        landmarks_signal.append([0, (math.dist(p, q) / normalization_factor)])

        # Store the knee and nose landmarks
        knee_landmarks.append([p, q])
        nose_landmarks.append(q)

    # Process the signal of interest and apply a low-pass filter
    signalOfInterest = np.array(landmarks_signal)[:, 1]
    signalOfInterest = filterSignal(signalOfInterest, cutOffFrequency=7.5)

    # Resample the signal
    currentFs = 1 / fps
    desiredFs = 1 / 60
    duration = end_time - start_time

    timeVector = np.linspace(0, duration, int(duration / currentFs))
    newTimeVector = np.linspace(0, duration, int(duration / desiredFs))
    upsampleSignal = signal.resample(signalOfInterest, len(newTimeVector))

    # Find peaks in the resampled signal using a custom algorithm
    distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity = peakFinder(
        upsampleSignal, fs=60, minDistance=3, cutOffFrequency=7.5, prct=0.05
    )

    # Generate time vectors for plotting
    line_time = []
    sizeOfDist = len(distance)
    for index, item in enumerate(distance):
        line_time.append((index / sizeOfDist) * duration + start_time)

    # Initialize lists to store peaks and valleys data
    line_peaks = []
    line_peaks_time = []
    line_valleys_start = []
    line_valleys_start_time = []
    line_valleys_end = []
    line_valleys_end_time = []
    line_valleys = []
    line_valleys_time = []

    # Process each peak to extract relevant data
    for index, item in enumerate(peaks):
        line_peaks.append(distance[item['peakIndex']])
        line_peaks_time.append((item['peakIndex'] / sizeOfDist) * duration + start_time)

        line_valleys_start.append(distance[item['openingValleyIndex']])
        line_valleys_start_time.append((item['openingValleyIndex'] / sizeOfDist) * duration + start_time)

        line_valleys_end.append(distance[item['closingValleyIndex']])
        line_valleys_end_time.append((item['closingValleyIndex'] / sizeOfDist) * duration + start_time)

        line_valleys.append(distance[item['openingValleyIndex']])
        line_valleys_time.append((item['openingValleyIndex'] / sizeOfDist) * duration + start_time)

    # Initialize lists to store various metrics
    amplitude = []
    peakTime = []
    rmsVelocity = []
    maxOpeningSpeed = []
    maxClosingSpeed = []
    averageOpeningSpeed = []
    averageClosingSpeed = []

    for idx, peak in enumerate(peaks):
        # Calculate height measures for the peaks
        x1 = peak['openingValleyIndex']
        y1 = distance[peak['openingValleyIndex']]
        x2 = peak['closingValleyIndex']
        y2 = distance[peak['closingValleyIndex']]
        x = peak['peakIndex']
        y = distance[peak['peakIndex']]

        f = interpolate.interp1d(np.array([x1, x2]), np.array([y1, y2]))

        amplitude.append(y - f(x))

        # Calculate velocity measures
        rmsVelocity.append(np.sqrt(np.mean(velocity[peak['openingValleyIndex']:peak['closingValleyIndex']] ** 2)))
        averageOpeningSpeed.append((y - f(x)) / ((peak['peakIndex'] - peak['openingValleyIndex']) * (1 / 60)))
        averageClosingSpeed.append((y - f(x)) / ((peak['closingValleyIndex'] - peak['peakIndex']) * (1 / 60)))

        peakTime.append(peak['peakIndex'] * (1 / 60))

    # Calculate mean and standard deviation for amplitude and velocity
    meanAmplitude = np.mean(amplitude)
    stdAmplitude = np.std(amplitude)
    meanRMSVelocity = np.mean(rmsVelocity)
    stdRMSVelocity = np.std(rmsVelocity)
    meanAverageOpeningSpeed = np.mean(averageOpeningSpeed)
    stdAverageOpeningSpeed = np.std(averageOpeningSpeed)
    meanAverageClosingSpeed = np.mean(averageClosingSpeed)
    stdAverageClosingSpeed = np.std(averageClosingSpeed)

    # Calculate cycle duration metrics
    meanCycleDuration = np.mean(np.diff(peakTime))
    stdCycleDuration = np.std(np.diff(peakTime))
    rangeCycleDuration = np.max(np.diff(peakTime)) - np.min(np.diff(peakTime))
    rate = len(peaks) / (peaks[-1]['closingValleyIndex'] - peaks[0]['openingValleyIndex']) / (1 / 60)

    # Calculate decay metrics
    earlyPeaks = peaks[:len(peaks) // 3]
    latePeaks = peaks[-len(peaks) // 3:]
    amplitudeDecay = np.mean(distance[:len(peaks) // 3]) / np.mean(distance[-len(peaks) // 3:])
    velocityDecay = np.sqrt(
        np.mean(velocity[earlyPeaks[0]['openingValleyIndex']:earlyPeaks[-1]['closingValleyIndex']] ** 2)
    ) / np.sqrt(np.mean(velocity[latePeaks[0]['openingValleyIndex']:latePeaks[-1]['closingValleyIndex']] ** 2))
    rateDecay = (len(earlyPeaks) / (
            (earlyPeaks[-1]['closingValleyIndex'] - earlyPeaks[0]['openingValleyIndex']) / (1 / 60))
    ) / (len(latePeaks) / (
            (latePeaks[-1]['closingValleyIndex'] - latePeaks[0]['openingValleyIndex']) / (1 / 60)))

    # Calculate coefficients of variation
    cvAmplitude = stdAmplitude / meanAmplitude
    cvCycleDuration = stdCycleDuration / meanCycleDuration
    cvRMSVelocity = stdRMSVelocity / meanRMSVelocity
    cvAverageOpeningSpeed = stdAverageOpeningSpeed / meanAverageOpeningSpeed
    cvAverageClosingSpeed = stdAverageClosingSpeed / meanAverageClosingSpeed

    # Prepare the final JSON result
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

    # Serialize the final JSON result and write to a file
    json_object = json.dumps(jsonFinal, default=json_serialize)
    file_name = "finger_tap_left" if is_left_leg else "finger_tap_right"
    with open(file_name + ".json", "w") as outfile:
        outfile.write(json_object)

    return jsonFinal
