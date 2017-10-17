from django.db import models

# Create your models here.
class Imgs(models.Model):
	img = models.ImageField()

class Target(models.Model):
	orientation = models.CharField(max_length=10)
	shape = models.CharField(max_length=10)
	alphanumeric = models.CharField(max_length=10)
	color = models.CharField(max_length=10)
