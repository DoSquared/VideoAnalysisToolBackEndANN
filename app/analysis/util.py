import scipy.signal as signal  # Import signal processing functions from SciPy
import scipy.interpolate as interpolate  # Import interpolation functions from SciPy
import numpy as np  # Import NumPy for numerical operations
from app.analysis.finderPeaksSignal import peakFinder  # Import custom peak finding function

def filter_signal(raw_signal, fs=25, cut_off_frequency=5):
    """
    Filters the raw signal using a low-pass Butterworth filter.
    
    Parameters:
    raw_signal (array-like): The input raw signal.
    fs (int): The sampling frequency of the signal.
    cut_off_frequency (float): The cutoff frequency for the low-pass filter.
    
    Returns:
    array-like: The filtered signal.
    """
    # Design a 2nd order Butterworth low-pass filter
    b, a = signal.butter(2, cut_off_frequency, fs=fs, btype='low', analog=False)
    # Apply the filter to the raw signal
    return signal.filtfilt(b, a, raw_signal)

def get_output(up_sample_signal, duration, start_time):
    """
    Analyzes the up-sampled signal to find peaks, valleys, and various metrics.
    
    Parameters:
    up_sample_signal (array-like): The up-sampled signal.
    duration (float): The duration of the signal.
    start_time (float): The start time of the signal.
    
    Returns:
    dict: A dictionary containing analysis results and metrics.
    """
    # Use peakFinder to get distance, velocity, and peak-related indices
    distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity = peakFinder(
        up_sample_signal, fs=60, minDistance=3, cutOffFrequency=7.5, prct=0.05)

    line_time = []
    sizeOfDist = len(distance)
    
    # Generate time points corresponding to the distance data
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

    # Process peaks and valleys
    for index, item in enumerate(peaks):
        line_peaks.append(distance[item['peakIndex']])
        line_peaks_time.append((item['peakIndex'] / sizeOfDist) * duration + start_time)

        line_valleys_start.append(distance[item['openingValleyIndex']])
        line_valleys_start_time.append((item['openingValleyIndex'] / sizeOfDist) * duration + start_time)

        line_valleys_end.append(distance[item['closingValleyIndex']])
        line_valleys_end_time.append((item['closingValleyIndex'] / sizeOfDist) * duration + start_time)

        line_valleys.append(distance[item['openingValleyIndex']])
        line_valleys_time.append((item['openingValleyIndex'] / sizeOfDist) * duration + start_time)

    amplitude = []
    peakTime = []
    rmsVelocity = []
    averageOpeningSpeed = []
    averageClosingSpeed = []

    # Calculate metrics for each peak
    for idx, peak in enumerate(peaks):
        # Height measures
        x1 = peak['openingValleyIndex']
        y1 = distance[peak['openingValleyIndex']]

        x2 = peak['closingValleyIndex']
        y2 = distance[peak['closingValleyIndex']]

        x = peak['peakIndex']
        y = distance[peak['peakIndex']]

        # Interpolate the line connecting the opening and closing valleys
        f = interpolate.interp1d(np.array([x1, x2]), np.array([y1, y2]))

        # Calculate amplitude of the peak
        amplitude.append(y - f(x))

        # Calculate RMS velocity
        rmsVelocity.append(np.sqrt(np.mean(velocity[peak['openingValleyIndex']:peak['closingValleyIndex']] ** 2)))

        # Calculate average opening and closing speeds
        averageOpeningSpeed.append((y - f(x)) / ((peak['peakIndex'] - peak['openingValleyIndex']) * (1 / 60)))
        averageClosingSpeed.append((y - f(x)) / ((peak['closingValleyIndex'] - peak['peakIndex']) * (1 / 60)))

        # Timing of the peak
        peakTime.append(peak['peakIndex'] * (1 / 60))

    # Calculate means and standard deviations of various metrics
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

    # Calculate early and late peak metrics for decay analysis
    earlyPeaks = peaks[:len(peaks) // 3]
    latePeaks = peaks[-len(peaks) // 3:]
    amplitudeDecay = np.mean(distance[:len(peaks) // 3]) / np.mean(distance[-len(peaks) // 3:])
    velocityDecay = np.sqrt(np.mean(velocity[earlyPeaks[0]['openingValleyIndex']:earlyPeaks[-1]['closingValleyIndex']] ** 2)) / np.sqrt(np.mean(velocity[latePeaks[0]['openingValleyIndex']:latePeaks[-1]['closingValleyIndex']] ** 2))
    rateDecay = (len(earlyPeaks) / ((earlyPeaks[-1]['closingValleyIndex'] - earlyPeaks[0]['openingValleyIndex']) / (1 / 60))) / (len(latePeaks) / ((latePeaks[-1]['closingValleyIndex'] - latePeaks[0]['openingValleyIndex']) / (1 / 60)))

    # Calculate coefficients of variation
    cvAmplitude = stdAmplitude / meanAmplitude
    cvCycleDuration = stdCycleDuration / meanCycleDuration
    cvRMSVelocity = stdRMSVelocity / meanRMSVelocity
    cvAverageOpeningSpeed = stdAverageOpeningSpeed / meanAverageOpeningSpeed
    cvAverageClosingSpeed = stdAverageClosingSpeed / meanAverageClosingSpeed

    # Create final JSON object with all metrics
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
    }

    # Return the final JSON object containing all the analysis results
    return jsonFinal
