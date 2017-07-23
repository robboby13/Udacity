import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image and print out some stats
image = mpimg.imread('Sample Images/detroit1.jpg')
print('This image is: ',type(image),'with dimensions:', image.shape)

# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)
line_image = np.copy(image)

# When defining these pixel values I took the average of three RGB points (near, medium, far)
# Yellow: [157,127,102]   White: [176,165,162]
# Average of yellow and white: [167,146,132]
# Yellow had the best results reference images
red_threshold = 157
green_threshold = 127
blue_threshold = 102
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

#Since the image was take with the dashboard in it we had to move the vertices up
left_bottom = [680, 690]
right_bottom = [1190, 690]
apex = [890, 520]

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Identify pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) \
            | (image[:,:,1] < rgb_threshold[1]) \
            | (image[:,:,2] < rgb_threshold[2])

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

#Mask color selection
color_select[color_thresholds] = [0,0,0]

# Find where image is both colored right and in the region
line_image[~color_thresholds & region_thresholds] = [255,0,0]

# Display the image                 
plt.imshow(color_select)
plt.imshow(line_image)
plt.show()
