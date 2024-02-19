import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras_cv
import keras
from keras.losses import KLDivergence
from keras.utils import plot_model
from keras import backend

from configurations import classes
from configurations import preset_with_weights
from keras.models import Model

preset = "efficientnetv2_b2_imagenet"

def model(preset:str, num_class:int)->Model:
    if preset_with_weights not in preset_with_weights:
       raise ValueError(f'{preset} we want a pretrained model')


       # Build Classifier
    model = keras_cv.models.ImageClassifier.from_preset(preset=preset,
                                                        num_classes=len(classes)
                                                        )

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=  KLDivergence)

    return model

model=model(preset, classes)


# Model Sumamry
model.summary()