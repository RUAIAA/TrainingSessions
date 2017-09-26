from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

from .models import Imgs

# Create your views here.

def index(request):
	img = Imgs.objects.all().order_by('-pk')[0]
	return render(request, "viewer/index.html", {'image': img})

def submit(request):
	return render(request, "viewer/submit.html")

@api_view(['POST'])
def submitImage(request):
	img1 = request.data.get("myimg")
	obj = Imgs(img=img1)
	obj.save()

	return Response(status=status.HTTP_200_OK) 
