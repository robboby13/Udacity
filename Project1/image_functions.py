# Import packages necessary for the computations
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import sys
import math

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

def read_image(filepath):
    """
    Accepts images of type jpeg and png
    :param filepath: path to desired image
    :return: image in 0, 255 byte scale
    """
    extentsion = filepath.split(".")[-1]

    if str(extentsion).upper() in ("JPG", "JPEG"):
        return mpimg.imread(filepath)
    elif str(extentsion).upper() == "PNG":
        return (mpimg.imread(filepath)*255).astype('uint8')
    else:
        sys.exit("Unknown file type - try jpg and png files")

def grayscale(img):
    """
    Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')
    :param img: image to be converted to grayscale
    :return: image in grayscale
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def hsvscale(img):
    """
    Applies the HSV transform
    :param img: image to be converted to HSV scale
    :return: transformed image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def hslscale(img):
    """
    Applies the HSL transform
    :param img: image to be converted to HSL scale
    :return: transformed image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSL)

def hsv_mask(img):
    """
    Selects only green and white regions in the image after applying HSV transform
    :param img: image for masking
    :return: maked image
    """
    
    # Define the lower and upper bounds for the white color
    white_lwr = np.array([0, 0, 200])
    white_upr = np.array([180, 255, 255])
    
    # Define the lower and upper bounds for the yellow color
    yellow_lwr = np.array([20, 100, 100])
    yellow_upr = np.array([30, 255, 255])
    
    # Convert the scale from BGR to HSV
    hsv_img = hsvscale(img)
    
    # Get the white color mask
    white_mask = cv2.inRange(hsv_img, white_lwr, white_upr)


    # Get the yellow color mask
    yellow_mask = cv2.inRange(hsv_img, yellow_lwr, yellow_upr)
    
    # Combine two masks
    mask_combined = white_mask | yellow_mask

    # Use bitwise_and to mask the original image
    return cv2.bitwise_and(img, img, mask=mask_combined)

def gaussian_blur(img, kernel_size):
    """
    Applies a Gaussian kernel to reduce the noise in the image
    :param img: image for smoothing
    :param kernel_size: kernel size to be applied for the gaussian 
    :return: smoothed image
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    """
    Applies the Canny transform
    :param img: image 
    :param low_threshold: lower bound for signal rejection
    :param high_threshold: upper bound for detecting strong edges 
    :return: transformed image
    """
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    :param img: image 
    :param vertices: vertices for the polygon
    :return: masked image
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    Draws line segments onto image
    This function modifies the given image and draws lines on it. 
    The lines are based on the coordinates of 2 points
    :param img: image 
    :param lines: end point pairs for all line segments
    :param color: desired line color
    :param thickness: desired line thickness
    :return: image with lines
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_lines2(img, lines, min_y, weight, color=[[255, 0, 0], [0, 255, 0]], thickness = 15):
    """
    Draws left and right lane lines
    This function modifies the given image and draws lines on it. 
    The lines are based on the coordinates of 2 points
    :param img: image 
    :param lines: end point pairs for all line segments
    :param min_y: minimum y value for extending lines
    :param weight: weight factor for frame to frame smoothing
    :param color: desired line color
    :param thickness: desired line thickness
    :return: image with lines
    """
    
    # Global variable to store the information of previous frame 
    global prev_lines, is_video_file
    
    # Go through each line segment and classify them as right and left lines based on their slode
    # Store slope, intercept and length for each line segment
    left_line = []
    right_line = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if (0.4 < slope < 1.0) or (-1.0 < slope < -0.4):
                intercept = y1 - slope * x1
                length = math.sqrt((y2 - y1)**2.0 + (x2 - x1)**2.0)
                if slope > 0 :
                    left_line.append((slope, intercept, length))
                elif slope < 0:
                    right_line.append((slope, intercept, length))
            else:
                pass
    
    # Find the average slope and intercept for each line
    # Use line lengths as the weights so that longer segments dominate the averages
    left_line = np.array(left_line)
    left_line_slope = np.average(left_line[:, 0], weights=left_line[:, 2])
    left_line_intercept = np.average(left_line[:, 1], weights=left_line[:, 2])

    right_line = np.array(right_line)
    right_line_slope = np.average(right_line[:, 0], weights=right_line[:, 2])
    right_line_intercept = np.average(right_line[:, 1], weights=right_line[:, 2])
    
    # When processing video files, there can be frame to frame jitter in the drawn lines 
    # To eliminate, we can use the line information from the previous frame and calculate
    # a weighted average line 
    # w= 0.5 will place equal importance to both results while w > 0.5 will increase the contribution coming from the previous frame
    if not prev_lines:
        pass
    else:
        left_line_slope = weight * prev_lines[0][0] + (1.0 - weight) * left_line_slope
        left_line_intercept = weight * prev_lines[0][1] + (1.0 - weight) * left_line_intercept
        right_line_slope = weight * prev_lines[1][0] + (1.0 - weight) * right_line_slope
        right_line_intercept = weight * prev_lines[1][1] + (1.0 - weight) * right_line_intercept
    
    if is_video_file:
        prev_lines= tuple(((left_line_slope, left_line_intercept), (right_line_slope, right_line_intercept)))
    
    
    # Now using the line slope and intercept we will draw line that extend between the top 
    # section of the mask (min_y) and bottom of the image (max y)
    img_shape = img.shape

    left_y1 = img_shape[0]
    left_x1 = int((left_y1 - left_line_intercept) / left_line_slope)

    left_y2 = min_y
    left_x2 = int((left_y2 - left_line_intercept) / left_line_slope)

    right_y1 = img_shape[0]
    right_x1 = int((right_y1 - right_line_intercept) / right_line_slope)

    right_y2 = min_y
    right_x2 = int((right_y2 - right_line_intercept) / right_line_slope)
    
    cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color[0], thickness)
    cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color[1], thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Calculates hough line segments and draws them one the image provided
    :param img: should be the output of a Canny transform.
    :param rho: radial distance from the center
    :param theta: rotation angle 
    :param threshold: threshold for intersection counts
    :param min_line_len: minimum line length limit
    :param max_line_gap: maximum allowed line gap in pixels
    :return: Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def hough_lines_smooth(img, rho, theta, threshold, min_line_len, max_line_gap, min_y, weight):
    """
    Calculates hough line segments and draws extrapolated lines one the image provided
    :param img: should be the output of a Canny transform.
    :param rho: radial distance from the center
    :param theta: rotation angle 
    :param threshold: threshold for intersection counts
    :param min_line_len: minimum line length limit
    :param max_line_gap: maximum allowed line gap in pixels
    :return: Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines2(line_img, lines, min_y, weight)
    return line_img

def weighted_img(img, initial_img, α=1.0, β=0.5, λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)



