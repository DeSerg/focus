from django.db import models
import string
import random
import os

def random_choice():
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choices(alphabet, k=8))

def images_path():
        return os.path.join(settings.LOCAL_FILE_DIR, 'images')

# Create your models here.
class Album(models.Model):
    name = models.CharField(max_length=20, default=random_choice())
    path = models.CharField(max_length=100)

    def photos_num(self):
        count = 0
        for name in os.listdir(self.path):
            if os.path.isfile(os.path.join(self.path, name)):
                count += 1
        return count


