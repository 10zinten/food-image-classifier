import os
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings

from .forms import ClassifierForm
from .models import Classifier

from scipy.misc import imread, imresize, imsave
import tensorflow as tf
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
    img_path = os.path.join(media_path, 'ngoenga.png')
    img = Image.open(img_path)
    img = np.array(img)

    # Load model and predict
    model_dir = os.path.join(os.path.dirname(settings.BASE_DIR), 'model/')
    model_meta_path = os.path.join(model_dir, 'Food-image-classifer-0.0001-2conv-model.meta')

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        w1 = sess.run('W_conv1:0')

    return HttpResponse(w1)
