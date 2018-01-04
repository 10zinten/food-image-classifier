import os
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings

from .forms import ClassifierForm
from .models import Classifier

from scipy.misc import imread, imresize, imsave
import numpy as np
from PIL import Image

def predict(request):
    form = ClassifierForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        m = Classifier()
        m.image = form.cleaned_data['image']
        print(type(form.cleaned_data['image']))
        m.save()

        # result = feedfowrd()
        return HttpResponse('image upload success')

    context = {
        "form": form,
    }
    return render(request, 'image_form.html', context)


def feed_forward(request):
    media_path = os.path.join(os.path.dirname(settings.BASE_DIR), 'media_cdn/images')
    img_path = os.path.join(media_path, 'ngoenga.png')
    img = Image.open(img_path)
    img = np.array(img)
    return HttpResponse(img)
