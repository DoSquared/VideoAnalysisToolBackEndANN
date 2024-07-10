from app.analysis.YOLOTracker import YOLOTracker  # Import the YOLOTracker class from the specified module
import json  # Import the json module to handle JSON operations
import time  # Import the time module to measure execution time

def write_output_to_file(output, file_path):
    """
    Write the given output dictionary to a file in JSON format.
    
    Parameters:
    output (dict): The dictionary to write to the file.
    file_path (str): The path to the file where the output will be written.
    """
    with open(file_path, 'w') as outfile:
        json.dump(output, outfile)  # Serialize the output dictionary as a JSON formatted stream to the file

# Measure the start time
start_time = time.time()

# Run the YOLOTracker with the specified video and model, and store the output in outputDict
ouputDict = YOLOTracker("rigidity_gaby.mp4", 'yolov8n.pt', '')

# Print the elapsed time since the start time
print("--- %s seconds ---" % (time.time() - start_time))

# Write the output dictionary to a file named 'output.json'
write_output_to_file(ouputDict, 'output.json')
