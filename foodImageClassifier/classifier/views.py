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
from keras import backend as K
import numpy as np
from PIL import Image
import wikipedia as wk
from bs4 import BeautifulSoup as bs

# Path to input image
media_path = os.path.join(os.path.dirname(settings.BASE_DIR), 'media_cdn/images')

# Model names
indian_model_name = 'FIC-In-C7-B32-E11'
western_model_name = 'FIC-ResNet-50-TL-Model'

# Class names
names = ['biryani', 'dosa', 'gulab jamun', 'jalebi', 'momo (food)', 'samosa', 'tandoori chicken']


# Model paths
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

graph = K.get_session().graph


def upload_img(request):

    form = ClassifierForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        m = Classifier()
        m.image = form.cleaned_data['image']
        print(type(form.cleaned_data['image']))
        m.save()

        # result = feedfowrd()
        return HttpResponseRedirect('/predict')

    context = {
        "form": form,
    }
    return render(request, 'indian_food.html', context)

def parse_ingredients(dish_name):

    # Extract dish page and convert into html
    dish = wk.page(dish_name)
    html = dish.html()

    # Parse html and extract ingredients
    soup = bs(html, 'html.parser')
    ingredient_table = soup.find_all('td', class_='ingredient')
    ingredients = ingredient_table[0].find_all('a')

    # store all ingredients and return
    ingredient_list = []

    if not ingredients:
        ing_string = ingredient_table[0].text
        ing_string = ing_string[1:]
        ing_string_list = ing_string.split('optional ingredients:')
        ingredients = ing_string_list[0].split(',') # [0] - main ingredients

        for ingredient in ingredients:
            ingredient_list.append(ingredient)

        return ingredient_list

    for ingredient in ingredients:
        ingredient_list.append(ingredient.string)

    return ingredient_list


def predict(request):

    # Preprocess image
    img_path = os.path.join(media_path, os.listdir(media_path)[0])
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(x.shape)

    # Prediction
    global graph
    with graph.as_default():
        preds = loaded_model.predict(x)

    # Extract name of dish and ingredients
    dish_name = names[np.argmax(preds)]
    context = {
        'dish_name': dish_name,
        'ingredients': parse_ingredients(dish_name)
    }

    print(context)

    return render(request, 'result.html', context)

def clean_up(request):
    # Delete image instance from model
    Classifier.objects.all().delete()

    # Delete image from media directory
    for img in os.listdir(media_path):
        os.remove(os.path.join(media_path, img))

    return HttpResponseRedirect('/')
