from django.conf.urls import include, url

from .views import *

urlpatterns = [
    url(r'^view/', index, name="index"),
    url(r'^submit/', submit, name="submit"),
    url(r'^submitImage/', submitImage, name="submit_image"),
    url(r'^submitToInterop/', submitToInterop, name="submit_to_interop"),
    url(r'^interop/', interop, name="interop"),

]
