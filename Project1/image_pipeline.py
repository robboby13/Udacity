import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import imageio
import image_functions
from moviepy.editor import VideoFileClip
from IPython.display import HTML

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

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
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



img = mpimg.imread('Sample Images/solidYellowLeft.jpg')
print('This image is:', type(img), 'with dimensions:', img.shape)

def image_pipeline(img):
    gray = grayscale(img)
    kernel_size=5
    blur_gray=gaussian_blur(gray,kernel_size)
    low_threshold = 38
    high_threshold = 145
    edges=canny(blur_gray, low_threshold, high_threshold)
    vertices = np.array([[(138,537),(460,316), (500,316), (920,537)]], dtype=np.int32)
    masked_edges=region_of_interest(edges,vertices)

    rho = 1
    theta = np.pi / 180
    threshold_hough = 20
    min_line_length = 7
    max_line_gap = 5
    line_image = hough_lines(masked_edges, rho, theta, threshold_hough, min_line_length, max_line_gap)
    line_edges = weighted_img(line_image, img)

    if not is_video_file:
        fig = plt.figure()
        plt.imshow(line_edges)
        plt.show()

    return line_edges


def run_images():
    """
    Read all the images in the folder, process, display and write them. 
    """
    for img_name in os.listdir("/home/nvidia/opencv/samples/python/Udacity/Project1/Sample Images/"):
        folder_name = "Sample Images"
        path = folder_name + "/" + img_name
        image = read_image(path)
        
        print(img_name.upper())
        
        result = image_pipeline(image)
        
        nm, ext = img_name.split(".")
        out_file_name = "Results/" + nm + "_out." + ext
        cv2.imwrite(out_file_name, cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
       

prev_lines = []
is_video_file = True
white_output = 'Results/challenge.mp4'
clip1 = VideoFileClip("/home/nvidia/opencv/samples/python/Udacity/Project1/Sample Videos/challenge.mp4")
white_clip = clip1.fl_image(image_pipeline)
white_clip.write_videofile(white_output, audio=False)

#run_images()
