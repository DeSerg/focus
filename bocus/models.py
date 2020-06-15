from django.db import models
import os

import bocus.extern as ext


def images_path():
        return os.path.join(settings.LOCAL_FILE_DIR, 'images')

# Create your models here.
class Album(models.Model):
    name = models.CharField(max_length=20)
    path = models.CharField(max_length=100)

    def photos_num(self):
        count = 0
        pkl_filepath = os.path.join(self.path, ext.PKL_FILENAME)
        for name in os.listdir(self.path):
            filepath = os.path.join(self.path, name)
            if os.path.isfile(filepath) and filepath != pkl_filepath:
                count += 1
        return count


