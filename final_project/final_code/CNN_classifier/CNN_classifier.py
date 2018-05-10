
# coding: utf-8

# In this notebook I construct a simple CNN for my LSST simulations. These are in the /data/lsst_mocks_single directory. 
# 
# First, I construct training and validation sets by shuffling/splitting the data into training and testing sets with minimal pre-processing. I then build a simple convolutional neural network. I then evaluate the algorithm.

# In[1]:


import sys
sys.path.append('..')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('pylab inline')


# In[2]:


n_images = 100 # choose total number of images
n_test = int(n_images/5) # some fiducial fraction of images that you test on
n_train = int(n_images)*2 # some fiducial number of images that you train on


# In[3]:


#import modules 

import numpy as np
import os
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

image_width = 45 # number of x-pixels, vary if different simulation set
image_height = 45 # number of y-pixels, vary if different simulation set

# create 2 3D empty arrays to append lensed and unlensed images to, respectively
images = np.empty(shape=[0,image_width,image_height]) 
images2 = np.empty(shape=[0,image_width,image_height])

# filename to lensed and unlensed outputs
# [ADJUST PATH ON YOUR MACHINE TO DOWNLOAD DIRECTORY]
filename = os.listdir(os.getcwd()+"/data_of_lsst/lsst_mocks_single/lensed_outputs/0")
filename2 = os.listdir(os.getcwd()+"/data_of_lsst/lsst_mocks_single/unlensed_outputs/0")

# Import images and ignore DS.store file, as this throws an error
for file in filename[:n_images]:
    if not file.startswith('.'):
        try:
            image_file = get_pkg_data_filename("data_of_lsst/lsst_mocks_single/lensed_outputs/0/" + file)
            image_data = fits.getdata(image_file, ext=0, ignore_missing_end=True, padding=False)
            images = np.concatenate((images, [image_data]))
        except OSError as err:
            print('This file sucks %s' % (image_file))

            
print ("Done !")

for file in filename2[:n_images]:
    if not file.startswith('.'):
        try:
            image_file = get_pkg_data_filename("data_of_lsst/lsst_mocks_single/unlensed_outputs/0/" + file)
            image_data = fits.getdata(image_file, ext=0, ignore_missing_end=True, padding=False)
            images2 = np.concatenate((images2, [image_data]))
        except OSError as err:
            print('This file sucks %s' % (image_file))

print ("Done !")

# append downloaded images to empty 3D arrays
lensed_output_0 = images
unlensed_output_0 = images2


# Creating training and validation sets

# In[4]:


# Remove 'NaN' from datasets, as some simulations included NaN variables
lensed_output_0 = np.where(np.isfinite(lensed_output_0), lensed_output_0, 0)
unlensed_output_0 = np.where(np.isfinite(unlensed_output_0), lensed_output_0, 0)


# In[5]:


import random 
from keras.utils import np_utils

# Set aside data for testing
# Concatenate lensed and unlensed systems to create an ndarray twice the size of n_images
training_data= np.concatenate((lensed_output_0, unlensed_output_0), axis=0)
training_labels = np.concatenate((np.ones(n_images), np.zeros(n_images)), axis=0)

# Combine, shuffle, and randomize data
combined = list(zip(training_data,training_labels))
random.shuffle(combined)
training_data, training_labels = zip(*combined)
training_data = np.array(training_data)
training_labels = np.array(training_labels)

# Create training set and labels using some portion of the data (can be adjusted as needed)
X_train = training_data[:int(n_images)*2]
y_train = training_labels[:int(n_images)*2]
X_test = training_data[1600:n_train] #this starting integer value should be adjusted for the n_images you include
y_test = training_labels[1600:n_train]


# In[6]:


# We use the full set for training
x = X_train.reshape((-1,1,45,45))
y = y_train.reshape((-1,1))
# We reuse training set as our validation set
xval = X_test.reshape((-1,1, 45, 45))
yval = y_test.reshape((-1,1))


# In[7]:


# Clipping and scaling parameters applied to the data as preprocessing
vmin=-1e-9
vmax=1e-9
scale=100

mask = np.where(x == 100)
mask_val = np.where(xval == 100)

x[mask] = 0
xval[mask_val] = 0

# Simple clipping and rescaling the images
x = np.clip(x, vmin, vmax)/vmax * scale
xval = np.clip(xval, vmin, vmax)/vmax * scale 

x[mask] = 0
xval[mask_val] = 0


# In[8]:


import matplotlib.pyplot as plt

#Illustration of a lens in one color band
im = x[0].T
plt.subplot(221)
plt.imshow(im[:,:,0]); plt.colorbar()


# # Classic CNN

# In[9]:


#import modules
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D


# In[10]:


nb_classes = 2

# input data dimensions
data_len = 45

# reshape X data
img_cols = 45
img_rows = 45
input_shape=(img_cols, img_rows, 1)

X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 1)
X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 1)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# In[68]:


# Sequential Model

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])


# In[74]:


# Train
batch_size = 194
nb_epoch = 30
model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=2, 
          validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Check that the training set and labels match up, and the accuracy of the image classification
# assignmnent

list(zip(model.predict(X_test[:20]).T[1], model.predict_classes(X_test[:20]), y_test[0:20]))


# Plot simulated images alongside their labels and classified labels to see how accurate 
# the model was


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
for i in range(len(y_test)):
    print(i)
    print(y_test[i])
    plt.imshow(X_test[i].reshape(45, 45))
    plt.show()


# I'm only getting ~50% accuracy. I think this is because some images are extrmely noisy--as we can see in the plots above, the noise may be diluting the sample. The simulations were produced over a variety of signal-to-noise and Einstein radii, which introduces noise that may be contaminating the sample. 
# 
# The next step would be to include a filtering method, where I include a threshold that marks whether something is a bad image or not.
# 

