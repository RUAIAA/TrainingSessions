from django.conf.urls import include, url

from .views import *

urlpatterns = [
    url(r'^view/', index, name="index"),
    url(r'^submit/', submit, name="submit"),
    url(r'^submitImage/', submitImage, name="submit_image"),
]
