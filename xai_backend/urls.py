from django.contrib import admin
from django.urls import path, include
from xai_app.views import home  # Import the home view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),  # Add this line for the home view
    path('api/', include('xai_app.urls')),  # API endpoint
]
