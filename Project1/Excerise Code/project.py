import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import imageio
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Read in image
image = mpimg.imread('Sample Images/solidYellowLeft.jpg')
print('This image is:', type(image), 'with dimensions:', image.shape)

#Convert to grayscale and add additional gaussian smoothing
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

#Apply Canny Edge Detection
# Define our parameters for Canny and apply
low_threshold = 38
high_threshold = 145
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

#Masked edges using cv2.fillpoly()
mask = np.zeros_like(edges)   
ignore_mask_color = 255 

# This time we are defining a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(138,537),(460,316), (500,316), (920,537)]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

# Define the Hough transform parameters
rho = 2
theta = np.pi/180
threshold = 12
min_line_length = 10
max_line_gap = 20
line_image = np.copy(image)*0 #creating a blank to draw lines on

# Run Hough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on the blank
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges)) 

# Draw the lines on the edge image
#combo = cv2.addWeighted(color_edges, 0.5, line_image, 1, 0)

#Draw the lines on the edge image
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 0.9, 0.5)

os.listdir("Sample Images/")
plt.imshow(lines_edges) 
plt.show()

