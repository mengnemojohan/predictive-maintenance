from django.urls import path, include

urlpatterns = [
    path('api/', include('motor_app.urls')),
]