import os
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings

from .forms import ClassifierForm
from .models import Classifier

from scipy.misc import imread, imresize, imsave
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from PIL import Image

# Path to image folder
media_path = os.path.join(settings.BASE_DIR, 'media_cdn/images')
indian_model_name = 'FIC-Indian-Dish-ResNet-50-Model'
western_model_name = 'FIC-Indian-Dish-ResNet-50-Model'


def upload_img(request):
    # Delete all existing images field and image from media directory
    # Classifier.objects.all().delete()
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


def predict(request):
    img_path = os.path.join(media_path, os.listdir(media_path)[0])
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(x.shape)

    # Load model and predict
    model_dir = os.path.join(os.path.dirname(settings.BASE_DIR), 'models', 'keras')
    model_arch_path = os.path.join(model_dir, indian_model_name + '.json')
    model_weight_path = os.path.join(model_dir, indian_model_name + '.h5')

    # load json and create model
    json_file = open(model_arch_path)
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(model_weight_path)
    print('Loaded model from disk')

    # Prediction
    preds = loaded_model.predict(x)

    return HttpResponse(preds)

def clean_up(request):
    # Delete image instance from model
    Classifier.objects.all().delete()

    # Delete image from media directory
    for img in os.listdir(media_path):
        os.remove(os.path.join(media_path, img))

    return HttpResponseRedirect('/')
