from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from .forms import PhotoForm
from .models import Photo
from django.http import JsonResponse
import io, base64
from PIL import Image
import numpy as np
import io
import mimetypes


def index(request):
    template = loader.get_template('gan/index.html')
    context = {'form': PhotoForm()}

    return HttpResponse(template.render(context, request))


def predict(request):
    if not request.method == 'POST':
        return redirect('gan:index')
    
    form = PhotoForm(request.POST, request.FILES)
    if not form.is_valid():
        raise ValueError('Formが不正です。')
    
    file_name = form.cleaned_data['image'].name
    print("file_name=", file_name)
    photo = Photo(image=form.cleaned_data['image'])

    
    translated_img = photo.predict() # (1, 128, 128, 3)
    #photo.save()
    translated_img = np.squeeze(translated_img, 0)  # (1, 128, 128, 3) -> (128, 128, 3)
    translated_img = 0.5 * translated_img + 0.5 # 0～1に変換
    pil_img = Image.fromarray((translated_img * 255).astype('uint8'), mode='RGB')
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())
    img_str = str(img_str)[2:-1]
    content_type=mimetypes.guess_type(file_name)[0]
    translated_img = 'data:' + content_type + ';base64,' + img_str

    d = {
        'img_str' : translated_img,
    }
    return JsonResponse(d)



    