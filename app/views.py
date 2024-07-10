# Import necessary libraries and modules
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import cv2
from django.core.files.storage import FileSystemStorage
import os
import uuid
from app.analysis.YOLOTracker import YOLOTracker
import time
import json
from app.leg_raise_2 import final_analysis, updatePeaksAndValleys, updateLandMarks
import traceback

def home(req):
    # Render the 'index.html' template when the home function is called
    return render(req, "index.html")
    # Alternatively, return a simple HttpResponse with "Hello world!"
    # return HttpResponse("<h1>Hello world!</h1>")


def analyse_video(path=None):
    # If no path is provided, return 0s
    if path is None:
        return 0, 0

    try:
        # Attempt to initialize video capture with the given path
        data = cv2.VideoCapture(path)
    except Exception as e:
        # Print error message if initialization fails and return 0s
        print(f"Error in initialising cv2 with the video path : {e}")
        return 0, 0, 0

    # Count the number of frames in the video
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    # Get the frames per second (fps) of the video
    fps = data.get(cv2.CAP_PROP_FPS)

    # If frames or fps is 0, return 0s
    if int(frames) == 0 or int(fps) == 0:
        return 0, 0, 0

    # Calculate the duration of the video in seconds
    seconds = round(frames / fps)

    return seconds, frames, fps


def analyse_video_frames(path=None):
    # If no path is provided, return an empty dictionary
    if path is None:
        return {}

    try:
        # Print message indicating the start of analysis
        print("analysis started")
        start_time = time.time()
        # Perform video analysis using YOLOTracker
        ouputDict = YOLOTracker(path, 'yolov8n.pt', '')
        # Print message indicating completion of analysis
        print("Analysis Done")
        print("--- %s seconds ---" % (time.time() - start_time))

        return ouputDict
    except Exception as e:
        # Print error message if video processing fails and return error dictionary
        print(f"Error in processing video : {e}")
        return {'error': str(e)}


def update_plot_data(json_data):
    try:
        # Print message indicating the start of plot update
        print("updating plot started")
        start_time = time.time()
        # Perform update on peaks and valleys using provided JSON data
        outputDict = updatePeaksAndValleys(json_data)
        # Print message indicating completion of plot update
        print("updating the plot is Done")
        print("--- %s seconds ---" % (time.time() - start_time))

        return outputDict
    except Exception as e:
        # Print error message if plot update fails and return error dictionary
        print(f"Error in processing update_plot_data : {e}")
        return {'error': str(e)}


def leg_analyse_video(json_data, path=None):
    # If no path is provided, return an empty dictionary
    if path is None:
        return {}

    try:
        # Print message indicating the start of leg analysis
        print("analysis started")
        start_time = time.time()
        # Perform final analysis using provided JSON data and video path
        outputDict = final_analysis(json_data, path)
        # Print message indicating completion of analysis
        print("Analysis Done")
        print("--- %s seconds ---" % (time.time() - start_time))

        return outputDict
    except Exception as e:
        # Print error message if video processing fails and return error dictionary
        print(f"Error in processing video : {e}")
        traceback.print_exc()  # Print the stack trace of the exception
        return {'error': str(e)}


def handle_upload(request):
    # Check if any files are uploaded
    if len(request.FILES) == 0:
        raise Exception("No files are uploaded")

    # Check if 'video' field is present in form-data
    if 'video' not in request.FILES:
        raise Exception("'video' field missing in form-data")

    # Get the uploaded video file
    video = request.FILES['video']
    # Get the root directory of the application
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))

    # Generate a unique file name for the uploaded video
    file_name = str(uuid.uuid4().hex[:15].upper()) + ".mp4"
    # Set the folder path for uploads
    folder_path = os.path.join(APP_ROOT, 'uploads')

    # Set the complete file path
    file_path = os.path.join(folder_path, file_name)
    # Save the uploaded video to the specified file path
    FileSystemStorage(folder_path).save(file_name, video)
    # Print message indicating the video is saved
    print("video saved")

    # Analyze the video frames
    val = analyse_video_frames(file_path)
    # Remove the saved video file after analysis
    os.remove(file_path)

    return val


def handle_upload2(request):
    # Check if any files are uploaded
    if len(request.FILES) == 0:
        raise Exception("No files are uploaded")

    # Check if 'video' field is present in form-data
    if 'video' not in request.FILES:
        raise Exception("'video' field missing in form-data")

    # Get the uploaded video file
    video = request.FILES['video']
    try:
        # Parse JSON data from request
        json_data = json.loads(request.POST['json_data'])
    except json.JSONDecodeError:
        # Raise exception if JSON data is invalid
        raise Exception("Invalid JSON data")

    # Get the root directory of the application
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))

    # Generate a unique file name for the uploaded video
    file_name = str(uuid.uuid4().hex[:15].upper()) + ".mp4"
    # Set the folder path for uploads
    folder_path = os.path.join(APP_ROOT, 'uploads')

    # Set the complete file path
    file_path = os.path.join(folder_path, file_name)
    # Save the uploaded video to the specified file path
    FileSystemStorage(folder_path).save(file_name, video)
    # Print message indicating the video is saved
    print("video saved")

    # Perform leg analysis on the video
    val = leg_analyse_video(json_data, file_path)
    # Remove the saved video file after analysis
    os.remove(file_path)

    return val


@api_view(['POST'])
def get_video_data(request):
    # Handle POST request to get video data
    if request.method == 'POST':
        output = handle_upload(request)

        return Response(output)


@api_view(['POST'])
def leg_raise_task(request):
    # Handle POST request to perform leg raise task
    if request.method == 'POST':
        output = handle_upload2(request)

        return Response(output)


@api_view(['POST'])
def updatePlotData(request):
    # Handle POST request to update plot data
    if request.method == 'POST':
        try:
            # Parse JSON data from request
            json_data = json.loads(request.POST['json_data'])
        except json.JSONDecodeError:
            # Raise exception if JSON data is invalid
            raise Exception("Invalid JSON data")

        # Update plot data with the parsed JSON data
        output = update_plot_data(json_data)

        return Response(output)


@api_view(['POST'])
def update_landmarks(request):
    # Handle POST request to update landmarks
    if request.method == 'POST':
        try:
            # Parse JSON data from request
            json_data = json.loads(request.POST['json_data'])
        except json.JSONDecodeError:
            # Raise exception if JSON data is invalid
            raise Exception("Invalid JSON data")

        # Update landmarks with the parsed JSON data
        output = updateLandMarks(json_data)

        return Response(output)
