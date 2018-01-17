import os
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings

from .forms import ClassifierForm
from .models import Classifier

from scipy.misc import imread, imresize, imsave
from keras.models import model_from_json
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from PIL import Image

def upload_img(request):
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
    media_path = os.path.join(os.path.dirname(settings.BASE_DIR), 'media_cdn/images')
    img_path = os.path.join(media_path, 'donut.jpg')
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(x.shape)

    # Load model and predict
    model_dir = os.path.dirname(settings.BASE_DIR)
    model_arch_path = os.path.join(model_dir, 'FIC-ResNet-50-TL-Model.json')
    model_weight_path = os.path.join(model_dir, 'FIC-ResNet-50-TL-Model.h5')

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
