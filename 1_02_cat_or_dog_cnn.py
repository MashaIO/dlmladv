#%%
from __future__ import absolute_import, division, print_function

#%%
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback

#%%
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
input_size = (128, 128)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), input_shape=(*input_size, 3), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

#%%
model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])

#%%
batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/Users/sheik/repo/learning/ai/dla2z/CNN/dataset/training_set', 
                                                 target_size = input_size,
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/Users/sheik/repo/learning/ai/dla2z/CNN/dataset/test_set',
                                            target_size = input_size,
                                            batch_size = batch_size,
                                            class_mode = 'binary')

                                            
#%%
class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1
 
    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'
#%%
history = LossHistory()
model.fit_generator(training_set,
                         steps_per_epoch = 8000/batch_size,
                         epochs = 90,
                         validation_data = test_set,
                         validation_steps = 2000/batch_size,
						 workers=12,
						 max_queue_size=100,
						 callbacks=[history])
#%%
from keras.preprocessing import image
test_image = image.load_img('/floyd/input/dataset/single_prediction/cat_or_dog_1.jpg', target_size = input_size)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
prediction = 'Not yet'

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction)                                      