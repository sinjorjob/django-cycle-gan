from django.db import models
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
from gan.network import CycleGAN
from django.conf import settings

import numpy as np
import tensorflow as tf
import io, base64

graph = tf.get_default_graph()


class Photo(models.Model):

    image = models.ImageField(upload_to='images/')


    def predict(self):

        global graph
        
        with graph.as_default():
            cycle_gan = CycleGAN()
            img_data = self.image.read()
            img_bin = io.BytesIO(img_data)
            image = Image.open(img_bin).convert("RGB")
            translated_img = cycle_gan.predict(image)

            return translated_img