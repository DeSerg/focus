import os
import random
import string
import ntpath

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, FileResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.generic.edit import FormView
from django.core.files.storage import default_storage

from .forms import UploadPhotosForm, RecognizeForm
from .models import Album

import bocus.extern as ext
import tempfile
from zipfile import ZipFile
from .recognize import RecognizeFace


def index(request):
    album_list = Album.objects.all()
    context = {'album_list': album_list}
    return render(request, 'bocus/index.html', context)

def random_choice():
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choices(alphabet, k=8))

def create_album(request):
    album = Album()
    album.name = random_choice()
    album.path = os.path.join(ext.IMAGES_DIR, album.name)
    os.mkdir(album.path)
    album.save()
    return HttpResponseRedirect(reverse('bocus:detail', args=(album.id,)))

def detail(request, album_id):
    album = Album.objects.get(pk=album_id)
    upload_photos_form = UploadPhotosForm()
    recognize_form = RecognizeForm()

    context = {'album': album,
            'photos_num': album.photos_num(),
            'upload_photos_form': upload_photos_form,
            'recognize_form': recognize_form}

    return render(request, 'bocus/album.html', context)

def upload_photos(request, album_id):
    album = Album.objects.get(pk=int(album_id))
    if request.method == 'POST':
        print('Files: {}'.format(request.FILES))
        form = UploadPhotosForm(request.POST, request.FILES)
        if form.is_valid():
            files = request.FILES.getlist('file_field')
            for f in files:
                image_path = os.path.join(album.path, f.name)
                print('Saving file {}'.format(image_path))
                with open(image_path, 'wb+') as destination:
                    for chunk in f.chunks():
                        destination.write(chunk)
            rf = RecognizeFace(album.path)
            rf.perform_clustering()

        else:
            print('Invalid form! {}'.format(form.errors))

    return HttpResponseRedirect(reverse('bocus:detail', args=(album.id,)))

def recognize(request, album_id):
    album = Album.objects.get(pk=int(album_id))

    if request.method != 'POST':
        return HttpResponseRedirect(reverse('bocus:detail', args=(album.id,)))

    face_img_file = tempfile.NamedTemporaryFile()
    face_img_path = face_img_file.name

    upl_f = request.FILES['file']
    for chunk in upl_f.chunks():
        face_img_file.write(chunk)

    recognizer = RecognizeFace(album.path)
    filepaths = recognizer.perform_recognition(face_img_path)

    with tempfile.TemporaryDirectory() as tmpdirname:
        archive_path = os.path.join(tmpdirname, 'recognized.zip')
        print('archive_path: ', archive_path)
        zipObj = ZipFile(archive_path, 'w')

        for filepath in filepaths:
            zipObj.write(filepath, ntpath.basename(filepath))

        zipObj.close()

        response = FileResponse(open(archive_path, 'rb'))
        return response

