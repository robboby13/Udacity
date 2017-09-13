## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


---
###Writeup / README
 
[project code](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier2.ipynb)

###Data Set Summary & Exploration


I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


Here is an exploratory visualization of the data set. It is a bar chart showing all 43 classes and the number of samples associated with each class.

![alt text][https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/images/training_sign_bar_chart.png]

Here you can view each classID with associated sign name.

![alt text][https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/images/classes.png}


When preparing the data for training it was required to convert the traffic sign images from RGB (32, 32, 3) to grayscale (32, 32, 1). To do this we used the openCV library with python3 bindings. Following this step the data was normalized. Both of these functions were input into the preprocess function resulting in the following.


![alt text][https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/images/processed_grayscaled_normalized.png]


My final model was based off the LeNet2 architecture and consisted of the following layers (as commented in the code)

Layer 1: Convolution: input(32, 32, 1) output(28, 28, 6)
Layer 1: Activation Function (Relu)
Layer 1: Pooling: input(28, 28, 6) output(14, 14, 6)

Layer 2: Convolution: input(14, 14, 6) output(10, 10, 16)
Layer 2: Activation (Relu)
Layer 2: Pooling: input(10, 10, 16) output(5, 5, 16)

Layer 3: Convolution: input(5, 5, 16) output(1, 1, 400)
Layer 3: Activation (Relu)
Layer 3: Flatten: input(5,5,16) output(400)
Layer 3: Flatten X: input(1,1,400) output(400)
Layer 3: Concat: input(400 + 400) output(800)
Layer 3: Dropout

Layer 4 (Fully Connected): input(800) output(43)


When the training the model the EPOCHS, BATCH_SIZE, and sigma were adjusted to reach the 93% compliance mark.

EPOCHS = 100
BATCH_SIZE = 80
mu = 0
sigma = 0.125
rate = 0.009

My final model results were:
* validation set accuracy of 93.1%
* test set accuracy of 91.3%

* What was the first architecture that was tried and why was it chosen?
Originally the LeNet1 architecture was tried with unsuccusful results

* What were some problems with the initial architecture?
Unable to reach the 93$ accuarcy requirement
 

All test model images can be view ![alt text][https://github.com/robboby13/Udacity/tree/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg]


Here are ten German traffic signs that I found on the web:

![alt text][https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/eight.jpg] ![alt text][https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/fifteen.jpg] ![alt text][https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/five.jpg] 
![alt text][https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/fourteen.jpg] ![alt text][https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/nine.jpg]
![alt text][https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/thirteen.jpg] ![alt text][https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/thirty.jpg] ![alt text][https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/three.jpg] 
![alt text][https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/twelve.jpg] ![alt text][https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/twentyfive.jpg]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### Requirements for Submission
Follow the instructions in the `Traffic_Sign_Classifier.ipynb` notebook and write the project report using the writeup template as a guide, `writeup_template.md`. Submit the project code and writeup document.
