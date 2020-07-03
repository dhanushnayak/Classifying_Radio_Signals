#!/usr/bin/env python
# coding: utf-8




from livelossplot.tf_keras import PlotLossesCallback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.metrics import confusion_matrix
from sklearn import metrics

import numpy as np
np.random.seed(42)
import warnings;warnings.simplefilter('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
print('Tensorflow version:', tf.__version__)




train_images = pd.read_csv("dataset/train/images.csv",header=None)


# In[3]:


train_labels = pd.read_csv("dataset/train/labels.csv",header=None)


# In[4]:


valid_images = pd.read_csv("dataset/validation/images.csv",header=None)


# In[5]:


valid_lables = pd.read_csv("dataset/validation/labels.csv",header=None)


# In[6]:


train_images.head()
train_labels.head(3)


# In[7]:


valid_lables.head(3)
valid_images.head(3)


# In[8]:


train_images.shape, train_labels.shape


# In[9]:


valid_images.shape, valid_lables.shape


# In[10]:


x_train = train_images.values.reshape(3200,64, 128, 1) # 3000 images of 64 * 128 size and itz is greyscaled images ie 1
x_val = valid_images.values.reshape(800, 64, 128, 1)

y_train = train_labels.values
y_val = valid_lables.values




plt.figure(0,figsize=(12,12))
for i in range(1,4):
    plt.subplot(1,3,i)
    img = np.squeeze(x_train[np.random.randint(0,3200)])
    plt.imshow(img,cmap='gray')
    plt.axis("off")
    


# In[12]:


plt.imshow(np.squeeze(x_train[3]),cmap='gray')


# In[ ]:





from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(horizontal_flip = True)
datagen_train.fit(x_train)
datagen_val = ImageDataGenerator(horizontal_flip=True)
datagen_val.fit(x_val)


# In[ ]:






from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint


# In[15]:


# Initialising the CNN
model = Sequential()
# 1st Convolution
model.add(Conv2D(32,(5,5),padding="same",input_shape=(64,128,1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# 2nd Convolution layer
model.add(Conv2D(64,(5,5),padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# Flattening
model.add(Flatten())
# Fully connected layer
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.4))
model.add(Dense(4, activation="softmax"))


# In[ ]:






# In[18]:


initial_learning = 0.005
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning,
    decay_steps=5,
    decay_rate=0.96,
    staircase=True
)
optimizer=Adam(learning_rate=lr_schedule)


# In[20]:


model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=['accuracy'])
model.summary()


# In[ ]:






# In[25]:


checkpoint = ModelCheckpoint('model_weight.h5',monitor='val_loss',save_weights_only=True,mode='min',verbose=1)
callbacks = [PlotLossesCallback(),checkpoint]
batch_size = 32
history = model.fit(
    datagen_train.flow(x_train,y_train,batch_size=batch_size,shuffle=True),
    steps_per_epoch = len(x_train)//batch_size,
    validation_data = datagen_val.flow(x_val,y_val,batch_size=batch_size,shuffle=True),
    validation_steps = len(x_val)//batch_size,
    epochs=12,
    callbacks=callbacks
)


# In[ ]:






# In[26]:


model.evaluate(x_val,y_val)


# In[28]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns

y_true = np.argmax(y_val,axis=1)
y_pred = np.argmax(model.predict(x_val),1)
print(metrics.classification_report(y_true,y_pred))


# In[29]:


print("Classification acc ; %0.6f"%metrics.accuracy_score(y_true,y_pred))


# In[17]:


labels = ["squiggle", "narrowband", "noise", "narrowbanddrd"]


# In[ ]:




