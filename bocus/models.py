from django.db import models

def images_path():
        return os.path.join(settings.LOCAL_FILE_DIR, 'images')

# Create your models here.
class Album(models.Model):
    path = models.FilePathField(path=images_path, allow_folders=True)

