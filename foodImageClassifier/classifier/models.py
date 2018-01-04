from django.db import models

class Classifier(models.Model):
    image = models.ImageField(upload_to='images/', default='images/None/no-img.jpg')
