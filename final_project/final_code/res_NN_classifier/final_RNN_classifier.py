# In this notebook I construct a deep residual network CNN for my LSST simulations. These are in the /data/lsst_mocks_single directory. 
# 
# First, I construct training and validation sets by shuffle/split the data into training and testing sets with minimal pre-processing. I then use a residual neural network built from Lasagne, Theano, and Keras models to construct the CNN. This is repliated from Lanusse et al. (2017), but run on a much smaller number of simulations due to time and GPU restrictions. I then evaluate the algorithm and print an ROC curve to evaluate my model's accuracy.

# In[1]:


import sys
sys.path.append('..')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('pylab inline')


# In[30]:


n_images = # choose total number of images
n_test = int(n_images/5) # some fiducial fraction of images that you test on
n_train = int(n_images)*2 # some fiducial number of images that you train on


# In[31]:


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
filename = os.listdir(os.getcwd()+"/data/lsst_mocks_single/lensed_outputs/0")
filename2 = os.listdir(os.getcwd()+"/data/lsst_mocks_single/unlensed_outputs/0")

# Import images and ignore DS.store file, as this throws an error
for file in filename[:n_images]:
    if not file.startswith('.'):
        try:
            image_file = get_pkg_data_filename("data/lsst_mocks_single/lensed_outputs/0/" + file)
            image_data = fits.getdata(image_file, ext=0, ignore_missing_end=True, padding=False)
            images = np.concatenate((images, [image_data]))
        except OSError as err:
            print('This file sucks %s' % (image_file))

            
print ("Done !")

for file in filename2[:n_images]:
    if not file.startswith('.'):
        try:
            image_file = get_pkg_data_filename("data/lsst_mocks_single/unlensed_outputs/0/" + file)
            image_data = fits.getdata(image_file, ext=0, ignore_missing_end=True, padding=False)
            images2 = np.concatenate((images2, [image_data]))
        except OSError as err:
            print('This file sucks %s' % (image_file))

print ("Done !")

# append downloaded images to empty 3D arrays
lensed_output_0 = images
unlensed_output_0 = images2


# Creating training and validation sets

# In[32]:


# Remove 'NaN' from datasets, as some simulations included NaN variables
lensed_output_0 = np.where(np.isfinite(lensed_output_0), lensed_output_0, 0)
unlensed_output_0 = np.where(np.isfinite(unlensed_output_0), lensed_output_0, 0)


# In[77]:


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


# In[34]:


# We use the full set for training
x = X_train.reshape((-1,1,45,45))
y = y_train.reshape((-1,1))
# We reuse training set as our validation set
xval = X_test.reshape((-1,1, 45, 45))
yval = y_test.reshape((-1,1))


# In[35]:


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


# In[36]:


import matplotlib.pyplot as plt

#Illustration of a lens in one color band
im = x[0].T
plt.subplot(221)
plt.imshow(im[:,:,0]); plt.colorbar()


# # Using the Residual Neural Network

# In[37]:


# In this section, we train the RNN on the dataset prepared above.

# Note that all the data-augmentation steps required to properly trained the model are performed online
# during training, the user does not need to augment the dataset himself

# Check that the path to the deeplens.resnet_classifier is set correctly on your machine, should you wish to run this notebook

from deeplens.resnet_classifier import deeplens_classifier
# reload(sys.modules['deeplens.resnet_classifier'])



# Below are the fiducial rates I ran for 2,000 images. These hyper-parameters need to be adjusted.
# If the RNN does not converge, try lowering the learning_rate / learning_rate_drop
# Also try adjusting bach_size
# Adjusting n_epochs can help you converge, i.e. number of passes through the whole training set

model = deeplens_classifier(learning_rate=0.0001,  # Initial learning rate
                          learning_rate_steps=3,  # Number of learning rate updates during training
                          learning_rate_drop=0.001, # Amount by which the learning rate is updated
                          batch_size=128,         # Size of the mini-batch
                          n_epochs=50)           # Number of epochs for training


# In[39]:


model.fit(x,y,xval,yval) # Train the model, the validation set is provided for evaluation of the model


# In[40]:


# Saving the model parameters
model.save('deeplens_params.npy')


# In[41]:


# Completeness and purity evaluated on the training set 
# Completness = TPR
# Purity = FPR
# See README for details on TPR / FPR definitions 
model.eval_purity_completeness(xval,yval)


# In[42]:


# Plot ROC curve on the training set 
tpr,fpr,th = model.eval_ROC(xval,yval)
plt.title('ROC on Training set')
plt.plot(fpr,tpr)
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.xlim(0,1.0); plt.ylim(0.0,1.)
plt.grid('on')


# In[43]:


# Obtain predicted probabilities for each image
p = model.predict_proba(xval)


# # Classic CNN

# In[66]:


import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D


# In[67]:


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


# In[75]:


list(zip(model.predict(X_test[:20]).T[1], model.predict_classes(X_test[:20]), y_test[0:20]))


# # Classify Training Set

# Test model on noiseless LSST data.

# In[ ]:


import numpy as np
import os

image_width = 45
image_height = 45

images = np.empty(shape=[0,image_width,image_height])

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename


filename = os.listdir(os.getcwd()+"/lsst_noiseless_single")

for file in filename[:n_images]:
    if not file.startswith('.'):
        try:
            image_file = get_pkg_data_filename("lsst_noiseless_single/" + file)
            image_data = fits.getdata(image_file, ext=0, ignore_missing_end=True, padding=False)
            images = np.concatenate((images, [image_data]))
        except OSError as err:
            print('This file sucks %s' % (image_file))

print ("Done !")


# The following can be used for external validation data. I didn't have a set of external data,
# but one could include their own validation data if desired.


from astropy.table import Table, vstack
import glob
import os

def classify(model, dataset_path,
                             vmin=-1e-9, vmax=1e-9, scale=100):
    """
    This function classifies the LSST simulations
    """
    n_directory = 10

    cats = []
    # First loop over the subdirectories
    for i in range(n_directory):

        sub_dir = os.path.join(dataset_path)

        # Load the images
        print("Loading images in " + sub_dir)
        
        image_width = 45
        image_height = 45

        cat = np.empty(shape=[0,image_width,image_height])

        filename = os.listdir(os.getcwd()+"[directory to validation data]")

        for file in filename[:n_images]:
            if not file.startswith('.'):
                try:
                    image_file = get_pkg_data_filename("[directory to validation data]" + file)
                    image_data = fits.getdata(image_file, ext=0, ignore_missing_end=True, padding=False)
                    cat = np.concatenate((cat, [image_data]))
                except OSError as err:
                    print('This file sucks %s' % (image_file))
                    
        # Apply preprocessing
        mask = np.where(cat == 100)

        cat[mask] = 0

        cat = np.clip(cat, vmin, vmax) / vmax * scale

        cat[mask] = 0

        # Classify
        print ("Classifying...")
        p = model.predict_proba(cat)
        cat['is_lens'] = p.squeeze()

        cats.append(cat)

    catalog = vstack(cats)
    return catalog


# In[ ]:



# Utility function to classify the challenge data with a given model
cat = classify(model, '"[directory to validation data]"') # Applies the same clipping 
                                                          # and normalisation as during training


from sklearn.metrics import roc_curve,roc_auc_score

# Compute the ROC curve
fpr_test,tpr_test,thc = roc_curve(cat['is_lens'], cat['prediction'])

plot(fpr_test,tpr_test,label='CMU DeepLens')
xlim(0,1)
ylim(0,1)
legend(loc=4)
xlabel('FPR')
ylabel('TPR')
title('ROC evaluated on Testing set')
grid('on')

# Get AUROC metric on the whole testing set
roc_auc_score(cat['is_lens'], cat['prediction'])

