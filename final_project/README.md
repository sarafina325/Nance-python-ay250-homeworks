FINAL PROJECT DESCRIPTION:

For my final project, I constructed and used a convolutional neural network to identify galaxy-galaxy strong lenses. I trained and validated my model on a set of LSST-like mock observations including a range of lensed systems of various sizes and signal-to-noise ratios. I tried two methods of building my classifier: the first was a classical CNN architecture, and the second was a deep residual network architecture. Since a classical CNN architecture is limited to approx 12 layers, I tried to implement a deep residual network (resnet) architecture, but with limited success. As a quick summary, the resnet convolutional layers employ shortcut connections between the input and output of a stack of a few convolution layers, essentially learning the difference between the input and outputs. 

For my resnet, I directly adopt the DeepLens model from Lanusse et al. (2017). The model described and used below corresponds to my fiducial choice of depth and complexity, which I found was suitable for the complexity of the task and the size of the training data. The input images are first processed by a convolution layer, which uses ELU activation and batch normalization. I simplified the problem by only using single-band images, but the code can handle multi-band images if I choose to train on those in the future. The images then go through a pre-activated bottleneck residual unit. The output is then processed through a layer with a sigmoid activation function, which outputs a probability between 0 and 1. 

I implemented my own fiducial pre-processing techniques to replicate the CMU paper; namely, clipping and scaling the images. This differs from the pre-processing outlined in the CMUDeepLens paper, as they found their pre-processing didn’t have a “significant impact” on the results if it was omitted. 

I trained my resnet model on several batches of images (i.e. 200 - 2,000) to see how the classification increased in accuracy as the image number increased. I varied the number of epochs (i.e. number of passes over the entire training set) and learning rate accordingly, as the models with more images required much more time. As a result, I decreased the number of epochs and increased the learning rate as I increased the number of images. I trained the network in mini-batches of 128 images using ADAM (Kingma & Ba, 2015). The fiducial learning rate is divided by 10 every 40 epochs (‘learning_rate_steps’ and ‘learning_rate_drop’, respectively). 

I used single-band simulated images, which may decrease the classification accuracy, but for the purposes of this project yielded reasonable classification probabilities. The simulations were obtained from the CMU DeepLens team (https://arxiv.org/pdf/1703.02642.pdf)  (Nan Li, specifically) which were built using LensPop and PICS. 
-	



RESULTS:

I ran into a few issues with the simulation set; namely that there were ‘NaN’’s included in the set that corrupted the input. After removing these, my results are split into the results for the classic CNN, and my results for the residual NN. 

Classic CNN: My 2-D CNN typically yields accuracies around 50% for a training set of 20,000 images, which is emphatically not a good classifier (essentially random). More than likely the classifier isn’t learning at a high enough rate to increase accuracy over time. I would wager that this is because the classic CNN isn’t wide/deep enough (not enough layers). In my residual NN, I increase the layering to 46 layers. The 2 dimensions may add complexity to this problem; it’s harder to learn over 2D rather than a 1D sample like, say, a light curve. Another reason that the CNN may be getting such low accuracy is because some images are extremely noisy--as we can see in the plots above, the noise may be diluting the sample. The simulations were produced over a variety of signal-to-noise and Einstein radii, which introduces noise that may be contaminating the sample. This may also contaminate the residual NN. 

The next step would be to include a filtering method, where I include a threshold that marks whether something is a bad image or not.

Residual NN: My 2-D residual NN yielded a completeness, or True Positive Rate (TPR), of ~61% for a total number of 2,000 images (1,000 lenses and 1,000 non-lenses). The TPR is calculated as taking the ratio of detected lenses to the total number of lenses. It also yielded a contamination rate, or False Positive Rate (FPR), of ~52% for a training set of 2,000 images. The FPR is the fraction of non-lens images incorrectly identified as lenses. Given these stats, my residual NN doesn’t appear to be much better than randomly selecting galaxies as lenses/non lenses (). Part of this may be due to the smaller size of my training set. The fact that these images are 2D made the training process excessively long (took ~4 hours to train 2,000 images). If I used all of the 20,000 simulated images, my accuracy almost certainly would’ve improved; though my hyper parameters would need to be adjusted accordingly. I also reused some fraction of my training set as my validation set, which may explain why my accuracy is low. 


DIRECTORY/FILE DESCRIPTIONS:

'''
data/
'''

This directory contains the files needed to train the CNN and RNN. Below is a description of everything in this directory.

lsst_mocks_single/: This directory contains simulations of images of the LSST best single epoch images. This is divided into two subdirs:

	/lensed_outputs/: where lensed images are saved
	/unlensed_outputs/: where unlensed images are saved

Each of these subdirs includes a /0/ subdirectory, which includes our training set.

I was only able to upload 100 of each type of image due to size constraints in the repo. Note that each image is a -tar.gz file. To unzip, do the following:
	' find . -type f -exec gunzip {} + ' 
in your terminal.

If you'd like access to the full suite of simulated images, proceed to the following link: 
	' http://portal.nersc.gov/project/hacc/nanli/lsst_sl_mocks/ '
and download ' lsst_sl_mocks/tar '

A warning that it takes some non-trivial time to download the FITS images into a working .py notebook, depending on the number of images that you download. 

'''
test_code/
'''

This directory contains notebooks used to test my CNN and RNN. Below is a description of everything in this directory. 

CNN_classifier.ipynb: This notebook contains my CNN architecture run on a variety of hyper parameters and image batches. 

CNN_classifier_200 images.ipynb: This notebook contains my test RNN architecture run on 200 images.

CNN_classifier_20,000 images.ipynb: This notebook contains my test RNN architecture run on 20,000 images.

RNN_classifier.ipynb: This notebook contains my test RNN architecture

RNN_classifier_200 images.ipynb: This notebook contains my test RNN architecture run on 200 images.

RNN_classifier_2,000 images.ipynb: This notebook contains my test RNN architecture run on 2,000 images.

*note that the hyper parameters for NNs between notebooks are likely different, as image batch changes between notebooks.

*note that to run the RNN_classifier, the rest of the deep learning architecture is in the final_code directory

'''
final_code/
'''

This directory contains the final code of the notebooks above. All of these can be run as demos (see below).

CNN_classifier/CNN_classifier.py: This file trains the CNN using the obtained data. It also prints the accuracy of the model and displays some plots of lensed galaxies predicted as lensed or unlensed.

res_NN_classifier/: This directory includes all of the .py modules needed to run the full residual neural network. To test, run: /final_RNN_classifier.py

'''
screenshots/
'''

Some screenshots of the code in action.

CNN_#imgs.png: These are screenshots of the outputs of my CNN_classifier.py on different numbers of training images. You can see the accuracy percentage as well as the number of epochs/time/training loss/value loss

CNN_#imgs.png: These are screenshots of the outputs of my res_NN_classifier.py on different numbers of training images. You can see the purity and completeness, as well as a ROC plot.

CNN_accurate_lens.png: This is a screenshot of my CNN in action, plotting a simulated galaxy lens and accurately predicting that it’s a lensed system.

CNN_inacurate_lens.png: This is a screenshot of my CNN in action, plotting an unsimulated galaxy lens and inaccurately predicting that it’s a lensed system.

DEMO:

All of the python files in the final_code/ directory can be run as demos. Note that if you wish to run these, you'll need to have the following libraries installed:
•	Theano
•	Lasagne
•	Keras
•	scikit-learn
•	astropy
•	pyfits

Note that there are some version issues with Theano/Keras, so if you wish to run the .py files you will need to ensure that the versioning is up-to-date. 
