from django.urls import path
from . import views

urlpatterns = [
    path('upload_data', views.upload_data, name='upload_data'),
    path('get_latest_data/', views.get_latest_data, name='get_latest_data'),
]