#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd
from os import listdir
import cv2
from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
import imutils
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


TRAINING_DIR = (r"C:\Users\HP\Dataset Plant Disease\Veg-dataset\Veg-dataset\train_set")

train_datagen = ImageDataGenerator(rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))


# In[6]:


VALIDATION_DIR = (r"C:\Users\HP\Dataset Plant Disease\Veg-dataset\Veg-dataset\test_set")

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=100,
                                                              class_mode='binary',
                                                              target_size=(150, 150))


# In[7]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[8]:


from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2,     "This notebook requires TensorFlow 2.0 or above."


# In[9]:


import tensorboard
tensorboard.__version__


# In[17]:


#model building using MobileNetV2

from tensorflow.keras.applications import MobileNetV2
basemodel=MobileNetV2(weights="imagenet",include_top=False,
                      input_tensor=Input(shape=(224,224,3)))
model=basemodel.output
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(100, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax')
    
])
#model summary
print(basemodel.summary())


# In[18]:


(train_images, train_labels), _ = keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0


# In[20]:


logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Train the model.
opt=tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])


history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=3,
            verbose=1)


# In[21]:


model.evaluate(validation_generator)


# In[22]:


model.save('Vegetable_MobileNetV2.h5')


# In[23]:


from keras.preprocessing.image import load_img,img_to_array

dic=train_generator.class_indices

icd={k:v for v,k in dic.items()}

def output(location):

    img=load_img(location,target_size=(150,150,3))

    img=img_to_array(img)

    img=img/255

    img=np.expand_dims(img,[0])

    answer=model.predict_classes(img)

    probability=round(np.max(model.predict_proba(img)*100),2)
    print ("**",icd[answer[0]], 'With probability',probability,"\n")


# In[25]:


img=r"C:\Users\HP\Dataset Plant Disease\Veg-dataset\Veg-dataset\test_set\Tomato___Bacterial_spot\b29993ed-4d96-499b-ad36-379fda20c9c7___UF.GRC_BS_Lab Leaf 8843.JPG"
s=load_img(img,target_size=(150,150,3))
s


# In[26]:


output(img)


# In[29]:


img=r"C:\Users\HP\Dataset Plant Disease\Veg-dataset\Veg-dataset\test_set\Pepper,_bell___healthy\ba527e73-8be2-4c53-9e5b-cafeab987bba___JR_HL 8037.JPG"
s=load_img(img,target_size=(150,150,3))
s


# In[30]:


output(img)


# In[31]:


img=r"C:\Users\HP\Dataset Plant Disease\Veg-dataset\Veg-dataset\test_set\Potato___Early_blight\b7157976-61c2-4366-87c5-e3de23aa7c10___RS_Early.B 7227.JPG"
s=load_img(img,target_size=(150,150,3))
s


# In[32]:


output(img)


# In[34]:


acc      = history.history['acc' ]
val_acc  = history.history[ 'val_acc' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) 

plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()


# In[ ]:




