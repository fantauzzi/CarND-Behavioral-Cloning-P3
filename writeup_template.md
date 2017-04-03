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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I have adopted the neural network architecture developed by NVIDIA for end-to-end deep learning of self-driving cars, are reported in their [blog](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). 

Three cameras are mounted in the front of the car, with a visual angle roughly comparable to that of a driver facing forward. The model maps pixels from the camera mounted on the car to a steering angle to be applied for driving.

During training of the network, images from the cameras are fed into it, and its output, a single value, is compared to the angle actually steered by the driver, then the network weights are updated to minimise the mean square error between the two.

When the program drives, images from the center cameras are fed to the trained model, and the resulting steering angle is applied. Images from the side cameras are used for training but not for driving, as illustrated below.

####2. Attempts to reduce overfitting in the model

The model uses a stride of 2 in the first three convolutional layers, which reduces the number of trained parameters and the spatial overlapping of the convolution output, which should help control overfitting.

Trending of the computed loss during training on training and validation data were comparable, and didn't show signs of overfitting.

Most important, the network drove successfully around a track that was never used for training, therefor able to generalise from the received training data. 

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. However, I tuned the correction to the steering angle for the left and right camera, by trial and error.

####4. Appropriate training data

I adopted the training data provided by Udacity with the simulator; because they didn't cover certain relevant situations, I have complemented them by recording some additional data.  

###Model Architecture and Training Strategy

####1. Solution Design Approach

I first adopted a very simple network design, with a fully connected layer and the output layer, to see I could train the model and use it to drive the car in the simulator. I use training data Udacity provided along with the simulator, consisting in more than 8000 images coming from each of the three cameras on the car, for a total of more than 24000 images, along with recorded telemetry data. Training data were all recorded on the same track, that I refer to as the training track. The simulator also allows driving on a second track, which I refer to as the test track. 

An increase in model complexity, with a number of convolutional layers followed by fully connected layers, allowed the  drive the car for a portion of the training track, before going off-road. I decided to try the network architecture developed by NVIDIA, and also to augment the dataset in the way described in the paper. I use images captured from the lateral cameras,  
The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

Having adopted Udacity's dataset, and after augmentation and pre-processing of training data, the program was able to drive around the training track successfully. However, it would go off-road in the test track. The latter is narrower, has changes in elevation, tighter turns, and a lane marking along the middle. From the road it sometimes possible to see other portions of the track that are not accessible from there, but nevertheless can confond the program that tries to drive toward them. In screen capture here below some situations where the program went off-road or anyway collided with an obstacle. TODO 
 
![alt text][image2]

![alt text][image3]

 
Udacity's dataset records driving mostly in the middle of the road, while in the problematic spots of the test track the car was very close to the side, and at a steep angle with it. 
 
 Increasing the steering correction for side images allowed to go around some of the tight turns, but not all of them; also, the car would wobble and go off-road in straight sections. 
 
 I tried additional tactics for data augmentation. I shifted images down and to the sides, I rotated them, by a constant or random amound, along with a correction to the steering angle. I darkened them. I also tried different color spaces in pre-processing (LAB, HSV). For each of these tactics I tried a number of variations: different ways to fill in 'empty' pixels after roto-translation, different centers of rotation, etc.. It was time-consuming, and didn't bring about significant improvement on the test track.
 
 What proved effective, instead, was to collect additional training data from the training track: I placed the car close to the curb and at a steep angle, then steered hard and begun to accelerate before starting to record data, and then recorded while I was taking the car to drive along the middle of the road. A few of those, and additional around 800 images per camera, and the trained network drove around the test track multiple times without any accident, and after one hour of driving, the car was still going around.   

 Since I adopted NVIDIA's model, a few epochs were enough to train the network at its best: four or five. Going past five epochs, loss on validation data would stay almost constant, and driving behavior would not improve. Here below a chart of training and validation loss over a training of 20 epochs with the final network architecture and dataset.
![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
