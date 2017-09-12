import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

img = mpimg.imread('/home/dura/CarND-Traffic-Sign-Classifier-Project/test/8.png')
img1 = mpimg.imread('/home/dura/CarND-Traffic-Sign-Classifier-Project/test/13.png')
img2 = mpimg.imread('/home/dura/CarND-Traffic-Sign-Classifier-Project/test/14.png')
img3 = mpimg.imread('/home/dura/CarND-Traffic-Sign-Classifier-Project/test/25.png')
img4 = mpimg.imread('/home/dura/CarND-Traffic-Sign-Classifier-Project/test/30.png')
#lum_img = img[:, :, 0]
plt.figure(1)

plt.subplot(211)
plt.imshow(img)

plt.subplot(212)
plt.imshow(img1)

plt.subplot(121)
plt.imshow(img2)

plt.subplot(111)
plt.imshow(img3)

plt.subplot(111)
plt.imshow(img4)

plt.show()
