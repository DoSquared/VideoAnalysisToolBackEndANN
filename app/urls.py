from django.urls import path  # Import the path function from Django's urls module
from app.views import home, get_video_data, leg_raise_task, updatePlotData, update_landmarks  # Import view functions from the app.views module

# Define URL patterns and map them to view functions
urlpatterns = [
    path('', home),  # Root URL, maps to the home view
    path('video/', get_video_data),  # URL for video data, maps to the get_video_data view
    path('leg_raise/', leg_raise_task),  # URL for leg raise task, maps to the leg_raise_task view
    path('update_plot/', updatePlotData),  # URL to update plot data, maps to the updatePlotData view
    path('update_landmarks/', update_landmarks),  # URL to update landmarks, maps to the update_landmarks view
    path('toe_tap/', leg_raise_task)  # URL for toe tap task, maps to the leg_raise_task view (reused)
]
