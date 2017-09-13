#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
Juniper Notebook [project code](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier2.ipynb)


I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


Here is an exploratory visualization of the data set. It is a bar chart showing all 43 classes and the number of samples associated with each class.

![alt text](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/images/training_sign_bar_chart.png)


Each classID had an appropriate label for their respective German Traffic Sig

![alt text](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/images/classes.png)



When preparing the data for training it was required to convert the traffic sign images from RGB (32, 32, 3) to grayscale (32, 32, 1). To do this we used the openCV library with python3 bindings. Following this step the data was normalized. Both of these functions were input into the preprocess function resulting in the following.


![alt text](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/images/processed_grayscaled_normalized.png)



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


When training the model EPOCHS, BATCH_SIZE, and sigma were adjusted to reach a 93% accuaracy mark.

My final model results were:
* validation set accuracy of 93.1%
* test set accuracy of 91.3%

Final Model Parameters
EPOCHS = 100
BATCH_SIZE = 80
mu = 0
sigma = 0.125
rate = 0.0009

An iterative approach was taken, originally the LeNet1 architecture was tried with unsuccusful results

With the LeNet1 Architecture I was unable to meet the 93% accuarcy requirement


Here are ten German traffic signs that I found on the web, the preprocessing function was applied to all ten for grayscale and data normalization.

![Class ID 5](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/five.jpg)
![Class ID 8](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/eight.jpg)
![alt text](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/twelve.jpg)
![alt text](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/nine.jpg)
![alt text](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/thirteen.jpg)
![alt text](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/three.jpg)
![Class ID 15](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/fifteen.jpg) 
![alt text](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/thirty.jpg)
![Class ID 14](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/fourteen.jpg) 
![alt text](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/twentyfive.jpg)


Here are the results of the predictions:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 80 KpH      		| Dangerous Curve to the right   									| 
| Speed Limit 120 Kph     			| Speed Limit 120 KpH 										|
| Priority Road					| Priority Road											|
| No Passing	      		| End of no passing					 				|
| Yield			| Yield      							|
| Speed Limit 60 KpH			| Priority Road      							|
| No Vehicles			| Speed Limit 30 KpH      							|
| Beware of Ice and Snow			| Bicycles crossing      							|
| Stop			| Speed Limit 30 KpH      							|
| Road Work			| Bicycles crossing      							|


The model was only 30% accuarate! NOT GOOD. This may have to do with the images not being fully centered, some of the images have significant amount of noise. 


For example the stop sign is not completley centered.

![Uncentered STOP Sign](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/fourteen.jpg) 

Below you can find the top 5 predictions for each German Traffic Sign

![Class ID 5](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/five.jpg)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .209         			| Dangerous Curve to the right   									| 
| .121     				| General caution 										|
| .0915					| Stop											|
| .0704	      			| Right-of-way at the next intersection					 				|
| .064				    | Traffic signals      							|


![Class ID 8](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/eight.jpg)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .128         			| Speed limit (120km/h)   									| 
| .106     				| Speed limit (80km/h) 										|
| .0732					| Speed limit (30km/h)											|
| .0627	      			| Speed limit (20km/h)					 				|
| .0491				    | Speed limit (70km/h)      							|

![alt text](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/twelve.jpg)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .126         			| Priority road   									| 
| .0823     				| Roundabout mandatory 										|
| .0615					| Stop											|
| .0552	      			| Road work					 				|
| .0429				    | Turn right ahead      							|

![alt text](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/nine.jpg)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0949         			| End of no passing   									| 
| .0559     				| No passing 										|
| .0488					| End of no passing by vehicles over 3.5 metric tons											|
| .0483	      			| Right-of-way at the next intersection					 				|
| .0411				    | No passing for vehicles over 3.5 metric tons      							|

![alt text](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/thirteen.jpg)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0959         			| Yield   									| 
| .0509     				| Priority road 										|
| .0383					| End of all speed and passing limits											|
| .0365	      			| Speed limit (30km/h)					 				|
| .034				    | Roundabout mandatory      							|

![alt text](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/three.jpg)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .0123         			| Priority road   									| 
| .0114     				| Speed limit (60km/h) 										|
| .0827					| Speed limit (80km/h)											|
| .0708	      			| Speed limit (50km/h)					 				|
| .0524				    | Children crossing      							|


![Class ID 15](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/fifteen.jpg) 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .298         			| Speed limit (30km/h)   									| 
| .163     				| Traffic signals 										|
| .123					| Keep left											|
| .117	      			| No entry					 				|
| .0759				    | Go straight or left      							|

![alt text](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/thirty.jpg)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .2         			| Bicycles crossing   									| 
| .194     				| Speed limit (80km/h)										|
| .146					| Bumpy road											|
| .0583	      			| Roundabout mandatory					 				|
| .0571				    | Road work      							|

![Class ID 14](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/fourteen.jpg) 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .207         			| Speed limit (30km/h)   									| 
| .116     				| Speed limit (80km/h)										|
| .062					| Stop											|
| .0567	      			| Keep right				 				|
| .0487				    | Roundabout mandatory      							|

![alt text](https://github.com/robboby13/Udacity/blob/master/CarND-Traffic-Sign-Classifier-Project/test_images_jpg/twentyfive.jpg)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .137         			| Bicycles crossing   									| 
| .126     				| Priority road										|
| .073					| Speed limit (80km/h)											|
| .0611	      			| Ahead only				 				|
| .0503				    | Road work      							|
