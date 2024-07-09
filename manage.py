#This script sets up the environment for a Django project by specifying the settings module and then executes Django's command-line utility to handle administrative tasks based on the provided command-line arguments. It ensures that Django is properly imported and provides an error message if it is not.

#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
#sys module is imported to access command-line arguments and system
import sys


def main():
    """Run administrative tasks."""
    
    # Setting the default Django settings module for the 'backend' project
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
    
    try:
        # Trying to import the Django execute_from_command_line function
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        # If the import fails, raise an ImportError with error message 
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    
    # Execute the command-line utility with the provided arguments
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
