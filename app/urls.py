# ragapi/urls.py
from django.urls import path
from .views import hackrx_webhook

urlpatterns = [
    path('api/v1/hackrx/run', hackrx_webhook),
]
