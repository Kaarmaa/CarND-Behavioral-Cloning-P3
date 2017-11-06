#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Center_Basic.jpg "Basic Track - Center"
[image2]: ./examples/Center_Advanced.png "Advanced Track - Center"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Normal Image"
[image5]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run1.mp4 containing a captured lap around the basic track
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model (NVIDIA Architecture), and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a Keras implementation of the NVIDIA convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 68-72) 

The model includes:
- Cropping Layer (model.py lines 66)
- Data normalization layer using a Keras lambda layer (model.py lines 67)
- 5 Conv2D layers including RELU layers to introduce nonlinearity for all convolution layers (model.py lines 68-72),
- 4 Fully connected layers (model.py lines 73 - 77) 

####2. Attempts to reduce overfitting in the model

The model contained dropout layers in order to reduce overfitting, but they were removed for degraded performance compared to the delivered model. This held true for both tracks available.   (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 50-53). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 79).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of:
 - Center lane driving (High Speed, Two Laps)
 - Recovery sscenarios from the left and right sides of the road
 - Normal speed lap on the advanced track
 - Slow speed lap for more samples on basic track

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to replicate the proved NVIDIA model.

I thought this model might be appropriate because it seemed to be one the most reasonable solutions when weighing complexity and results.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. In an attempt to combat any overfitting, I introduced
a few dropout layers between the dense layers. However, this proved to actually be detrimental to the total performance of the model.

The main issue encountered was the left turn after the bridge on the basic track. Without providing additional training cases, the model would prefer to go straight, as opposed to 
making the hard(ish) left turn.

At the end of training with the additional cases, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 65 - 77) consisted of a convolution neural network with the following layers and layer sizes:

-  Data Preprocessing (In: 160 x 320 x 3)
    - Cropping      (65 x 320 x 3)
    - Normalization (65 x 320 x 3)
- Convolutional 2D (31 x 158 x 24)
- Convolutional 2D (14 x 77 x 36)
- Convolutional 2D (5 x 37 x 48)
- Convolutional 2D (3 x 35 x 64)
- Convolutional 2D (1 x 33 x 64)
- Flatten 
- Fully Connected (100)
- Fully Connected (50)
- Fully Connected (10)
- Fully Connected (1)

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

Then I added a lap around the advanced track

![alt text][image2]

I then recorded the first track again, but at a slower host velocity to ensure higher quality data (easier control), as well as simply expanding the data set

To augment the data set, I also flipped images and steering angle in order to vary the data, and provide effectively 6x the data from just using the single center camera


After the collection process, I had 17,500 samples, which after L/C/R and Flips, becomes 105,000 training images. 

These images were then only preprocessed by cropping to the targeted area of interest, and fed into the normalization layer.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was around 3 as evidenced by the plateau of the training accuracy and loss over time.

I used an adam optimizer so that manually training the learning rate wasn't necessary.
