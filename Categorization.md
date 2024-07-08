# Project Structure

## VideoAnalysisToolBackend-main

### User Interaction
- `app/static`: Contains CSS, JavaScript, and other static assets.
- `app/templates`: HTML templates for rendering the user interface.

### Request Handling
- `app/urls.py`: URL routing for the app.
- `backend/urls.py`: URL routing for the backend.
- `app/views.py`: Handles requests and responses for the app.
- `backend/views.py`: Handles requests and responses for the backend.

### Processing
- `app/models.py`: Defines the database models.
- `app/hand_analysis.py`: Script for hand analysis.
- `app/leg_raise_2.py`: Script for leg raise analysis.
- `app/runner.py`: Possibly a script to run various analyses.
- `app/analysis`: Likely contains additional analysis-related scripts or modules.

### Database Interactions
- `db.sqlite3`: SQLite database file.
- `app/migrations`: Database migration files.

### Configuration and Setup
- `backend/settings.py`: Django settings file.
- `backend/asgi.py`: ASGI configuration.
- `backend/wsgi.py`: WSGI configuration.
- `manage.py`: Django management script.
- `requirements.txt`: List of Python dependencies.

### Documentation
- `README.md`: Project documentation.

### Model Files
- `app/yolov8n.pt`: YOLO model file for object detection.

### Cache
- `__pycache__`: Compiled Python files.

### Unneeded Files
- `.DS_Store`: macOS system file.
- `.gitignore`: Specifies intentionally untracked files to ignor
