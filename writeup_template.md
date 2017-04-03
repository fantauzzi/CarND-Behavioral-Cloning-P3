#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

A program learns to steer a car, in a driving simulator, looking at someone else's driving. Udacity provided the driving simulator, and also a training dataset, consisting in images recorded while driving in the simulator, and related telemetry (e.g., the steering angle). It is possible to collect additional data, driving in the simulator. 

The program processes the training dataset, and trains a neural network that allows the car to drive autonomously, not only around the track where the training was done, but also around an additional track not seen during training.

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [specifications](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each requirement in my implementation.  

Files and directories content
 - `readme.md` this file.
 - `model.py` program to train the neural network, and save it in a binary file.
 - `drive.py` program that reads a trained network from the binary file and drives the car autonomously in the simulator.
 - `model.h5` binary file with the trained neural network.
 - `video.py` program for video recording of driving in the simulator, provided by Udacity.
 - `video.mp4` a video clip showing the car driving autonomously around two different tracks in the simulator.
 - `dataset\` additional training data (in addition to Udacity's dataset)
 
 Udacity's dataset can be downloaded from here. It should be unzipped and saved in directory `dataset` 
 
 ---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode


####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I have adopted the neural network architecture developed by NVIDIA for end-to-end deep learning of self-driving cars, as reported in their [blog](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). For mode details, also see NVIDIA's [paper](https://arxiv.org/pdf/1604.07316v1.pdf).  

Three cameras are mounted in the front of the car, with a visual angle roughly comparable to that of a driver facing forward. The model maps pixels from the camera mounted on the car to a steering angle to be applied for driving.

During training of the network, images from the cameras are fed into it, and its output, a single value, is compared to the angle actually steered by the driver, then the network weights are updated to minimise the mean square error between the two.

When the program drives, images from the center cameras are fed to the trained model, and the resulting steering angle is applied. Images from the side cameras are used for training but not for driving, as detailed below.

####2. Attempts to reduce overfitting in the model

NVIDIA's model uses a stride of 2 in the first three convolutional layers, which reduces the number of trained parameters and the spatial overlapping of the convolution output, which should help control overfitting.

Trending of the computed loss on training and validation data were comparable, and didn't show signs of overfitting.

Most important, the program drove successfully around a track that was never used for training, generalising from the received training data. 

####3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually. However, I tuned the correction to the steering angle for the left and right camera, by trial and error. A too small value made data augmentation insufficient, and the program drove straight is some bends. A too large value made the car swing from side to side in straight segments, wider and wider, until it went off-road. 

####4. Appropriate training data

I adopted the training dataset provided by Udacity with the simulator. Because it didn't cover certain relevant situations, I have complemented it by recording some additional data. Specifically, I have been starting with the car pointing to vertical poles by the side of the track, at a steep angle with the curb, to then drive back toward the middle of the road. See picture below.
    
    TODO

###Model Architecture and Training Strategy

####1. Solution Design Approach

I first adopted a very simple network design, with a fully connected layer and the output layer, to see that I could train the model and use it to drive the car in the simulator. The dataset provided by Udacity consists in more than 8000 images coming from each of the three cameras on the car, for a total of more than 24000 images, along with recorded telemetry data. Training data were all recorded on the same track, that I refer to as the **training track**. The simulator also allows driving on a second track, which I refer to as the **test track**. 

An increase in model complexity, with a number of convolutional layers followed by fully connected layers, allowed to drive the car for a portion of the training track, before going off-road.

I decided to try the network architecture developed by NVIDIA, and also to augment the dataset in the way NVIDIA's paper describes. I used images captured from the lateral cameras,  feeding them to the network for training like if they were images of the center camera, and adjusting the steering angle accordingly; that is, to the right for images taken from the left camera, and vice versa for the right camera. The amount of adjustement is a constant that I tuned by trial and error, finally setting it to 0.5 degrees.

I also tried adopting well-established, pre-trained models, i.e. VGG-16, VGG-19 and ResNet50, via transfer learning. In every case I didn't get any improvement, compared to the driving behavior of NVIDIA's model. This can be because those models had a different purpose, to classify images from the ImageNet database, and transfer-learning wasn't appropriate. Those models would have required to be re-trained on the new dataset, or at least fine-tuned. Given they are computationally expensive to train, I staid with NVIDIA's network architecture.
    
####2. Final Model Architecture

This is the breakdown of NVIDIA's model, as I have adopted. Input images (after pre-processing) are color images with resolution 320 x 65 pixels (width x height). 



![alt text][image1]

####3. Creation of the Training Set & Training Process

Pre-processing applies these step to every image:
 - **Cropping** from 320 x 160 pixels to 320 x 65, removing 70 lines of pixels from the top (not relevant to driving, as often above the horizon), and 25 lines from the bottom (where the car's hood is in the way). Cropping is done before any other pre-processing, as it lowers the image resolution and therefore the computational cost of the remaining operations.
 - **Conversion to the YUV color space**. It gave better results in driving, compared to RGB. In early experiments it showed slightly better results than LAB and HSV.
 - **Normalisation**, mapping all pixel values from the range of integers [0, 255] to the range of real numbers [-1, 1]. There is no 0 centering.
 
To provide network training with more useful data, I augment the dataset after pre-processing like described in NVIDIA's paper. Every image taken by a lateral camera can be used for training like if taken by the center camera, adjusting its corresponding steering angle. I also flip every image, and use it with its inverted steering angle. For every pair image-steering in the dataset, six are actually used for training: the image from the center camera, the images from the two lateral cameras, and the flipper version of each. 
 
 
Having adopted Udacity's dataset, and after augmentation and pre-processing of training data, the program was able to drive around the training track successfully. However, it would go off-road in the test track. The latter is narrower, has changes in elevation, tighter turns, and a lane marking along the middle. From the road it sometimes possible to see other portions of the track that are not accessible from there, but nevertheless can confond the program that tries to drive toward them. In screen capture here below some situations where the program went off-road or anyway collided with an obstacle. TODO 
 
![alt text][image2]

![alt text][image3]

I tried additional tactics for data augmentation. I shifted images down and to the sides, I rotated them, by a constant or random amound, along with a correction to the steering angle. It was time-consuming, and didn't bring about significant improvements on the test track.
 
What worked instead was to recorded additional data driving in the simulator, again in the training track, starting close to the curb and at a sharp angle with it. With another 800 samples in the dataset, the trained network was able to drive also around the test tarck, that it had never seen during training. 
 
Charting the validation and test loss per epoch showed that the model learned what it could within 10 to 12 epochs. But in practice, training past 8 to 10 epochs brought about worse driving, with the car going off-road in both tracks. 
 