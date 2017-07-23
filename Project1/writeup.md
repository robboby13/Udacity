# Finding Lane Lines on the Road

Insert Photo here

# Project Excerises
For the first excercise of determining which pixels to use I took the average of three pixels and generated two values for yellow and white 
Yellow: []
White: []
Average: []
The following images represents the results from each RGB value.
Note: These videos were taken on Detroit Roads

Region of Interest was adjusted to get rid of dashboard in collected video. ROI = Triangle

left_bottom = []
right_bottom = []
apex = []

Insert Images here

For the second excercise I had to adjust the region of interest to be a four sided polygon since the video collected has the dash board in the images

left_bottom = []
left_top = []
right_top = []
right_bottom = []

Insert Images

# Image Pipeline Function.

As part of the description, explain how you modified the draw_lines() function.
My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I ....
In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...
If you'd like to include images to show how the pipeline works, here is how to include an image:

# Shortcomings

One potential shortcoming would be what would happen when ...
Another shortcoming could be ...

# Needed Improvements

A possible improvement would be to ...
Another potential improvement could be to ...
