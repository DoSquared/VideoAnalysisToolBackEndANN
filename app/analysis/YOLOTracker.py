from ultralytics import YOLO  # Import YOLO model from ultralytics
import cv2  # Import OpenCV for video processing
import torch  # Import PyTorch for device handling

def YOLOTracker(filePath, modelPath, device='cpu'):
    # Check if CUDA is available, otherwise use the specified device
    device = 'cuda' if torch.cuda.is_available() else device
    
    # Load the YOLO model from the given model path
    model = YOLO(modelPath)
    
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(filePath)

    # Initialize a list to store bounding boxes for each frame
    boundingBoxes = []
    
    # Initialize frame number
    frameNumber = 0

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        
        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, classes=[0], verbose=False, device=device)
            
            # Initialize a list to store the data for the current frame
            data = []

            # Check if there are any results and if bounding boxes and IDs are available
            if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                # Convert IDs and bounding boxes to numpy arrays
                ind = results[0].boxes.id.cpu().numpy().astype(int)
                box = results[0].boxes.xyxy.cpu().numpy().astype(int)
                
                # Loop through each detected object
                for i in range(len(ind)):
                    # Create a dictionary for each object
                    temp = dict()
                    temp['id'] = int(ind[i])
                    temp['x'] = int(box[i][0])
                    temp['y'] = int(box[i][1])
                    temp['width'] = int(box[i][2] - box[i][0])
                    temp['height'] = int(box[i][3] - box[i][1])
                    temp['Subject'] = False  # Custom field, can be used for additional labeling
                    data.append(temp)

            # Store the results for the current frame
            frameResults = {'frameNumber': frameNumber, 'data': data}
            boundingBoxes.append(frameResults)

        else:
            # Break the loop if the end of the video is reached or reading fails
            break

        # Increment the frame number
        frameNumber += 1

    # Create the output dictionary
    outputDictionary = dict()
    # Store the frames per second of the video
    outputDictionary['fps'] = cap.get(cv2.CAP_PROP_FPS)
    # Store the bounding boxes data
    outputDictionary['boundingBoxes'] = boundingBoxes
    
    # Release the video capture object
    cap.release()
    
    # Return the output dictionary
    return outputDictionary
