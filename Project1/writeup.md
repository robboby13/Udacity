# Finding Lane Lines on the Road

![alt text](https://github.com/robboby13/Udacity/blob/master/Project1/Image%20Results/Project/solidwhitecurve.png)

# Project Excerises
For the first excercise of determining which pixels to use, an average of three pixels was taken and generated two RGB values for yellow and white 

The following images represent the results from each RGB value.
Note: These videos were taken on Detroit Roads

Region of Interest was adjusted to get rid of dashboard in collected video. ROI = Triangle

![alt text](https://github.com/robboby13/Udacity/blob/master/Project1/Image%20Results/lane_lines_1/yellow_lane_ROI.png)


For the second excercise the region of interest was adjusted to be a four sided polygon.

![alt text](https://github.com/robboby13/Udacity/blob/master/Project1/Image%20Results/lane_lines_2/lane_lines_roi.png)

# Image Pipeline Function.

The pipline is referenced below, the provided videos/images were used. The original image was taken to a gray scale this was then taken and applied to a gaussian smoothing function to eliminate any noise. The canny edge function was then applied. Followed by definig our ROI with four vertices. Hough Transform was applied with the below values. All results can be seen below


Image Pipeline: image_pipeline.py

# Solid White Curve
![alt text](https://github.com/robboby13/Udacity/blob/master/Project1/Image%20Results/Project/solidwhitecurve.png)
# Solid White Right
![alt text](https://github.com/robboby13/Udacity/blob/master/Project1/Image%20Results/Project/solidwhiteright.png)
# Solid Yellow Curve
![alt text](https://github.com/robboby13/Udacity/blob/master/Project1/Image%20Results/Project/solidyellowcurve.png)
# Solid Yellow Curve2
![alt text](https://github.com/robboby13/Udacity/blob/master/Project1/Image%20Results/Project/solidyellowcurve2.png)
# Solid Yellow Left
![alt text](https://github.com/robboby13/Udacity/blob/master/Project1/Image%20Results/Project/solidyellowleft.png)
# Whtie Car Lane Switch
![alt text](https://github.com/robboby13/Udacity/blob/master/Project1/Image%20Results/Project/whitecarlaneswitch.png)

# Shortcomings

Currently there is some merging at further distances that need to get improved upon. Overall the performance in the near range was fairly accurate, long range still needs some work. 


# Needed Improvements

The challange video shadows proved to be difficult and could still use some improvements
