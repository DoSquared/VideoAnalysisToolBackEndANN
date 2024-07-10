from django.shortcuts import render


def home(request):
    """
    Handle the home page view.

    Parameters:
    request (HttpRequest): The request object used to generate this response.

    Returns:
    HttpResponse: The response object containing the rendered template.
    """
    # Render the 'index.html' template and return the HTTP response
    return render(request, template_name='index.html')
