from django.urls import path

from . import views

app_name = 'bocus'

urlpatterns = [
    path('', views.index, name='index'),
    path('create_album/', views.create_album, name='create_album'),
    path('<str:album_id>/', views.detail, name='detail'),
    path('upload_photos/<str:album_id>/', views.upload_photos, name='upload_photos'),
    path('recognize/<str:album_id>/', views.recognize, name='recognize'),
]

