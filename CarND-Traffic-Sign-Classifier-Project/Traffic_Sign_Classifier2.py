
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[1]:


# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "traffic-signs-data/train.p"
validation_file="traffic-signs-data/valid.p"
testing_file = "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_valid shape:", X_valid.shape)
print("y_valid shape:", y_valid.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[2]:


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = max(y_train) + 1

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[3]:


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
# Visualizations will be shown in the notebook.
get_ipython().magic('matplotlib inline')

label_names = pd.read_csv('signnames.csv')
matplotlib.style.use('fivethirtyeight')
fig_size = (12,9)
plt.rcParams["figure.figsize"] = fig_size

training_labels = pd.DataFrame({'label': y_train})
label_counts = training_labels['label'].value_counts().sort_index()
plot = label_counts.plot(kind='bar', rot=0)

plot.set_title('Training Sign Classes')
plot.set_ylabel('Number of Signs')
plot.set_xlabel('Class ID')
plt.show()
#plt.savefig('examples/visualization.png')
print(label_names.to_string(index=False))


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[4]:


import cv2

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def normalize(img, norm_type=cv2.NORM_MINMAX):
    normalized_image = img.copy()
    cv2.normalize(img, normalized_image, norm_type=norm_type, dtype=cv2.CV_32F)
    return normalized_image

def preprocess(img_matrix):
    return_matrix = []
    for image in img_matrix:
        gray_image = grayscale(image)
        normal_image = normalize(gray_image)        
        return_matrix.append(normal_image)
    return return_matrix

# presprocess the images
X_train_processed = preprocess(X_train)
X_valid_processed = preprocess(X_valid)
X_test_processed = preprocess(X_test)

assert n_train == len(X_train_processed)
assert n_validation == len(X_valid_processed)
assert n_test == len(X_test_processed)

fig, axarr = plt.subplots(2)
axarr[0].imshow(X_train[58])
axarr[0].set_title('Input Image')
axarr[0].axis('off')
axarr[1].imshow(X_train_processed[58], cmap='gray')
axarr[1].set_title('Processed Image')
axarr[1].axis('off')

plt.show()


# ### Model Architecture

# In[6]:


### Define your architecture here.
### Feel free to use as many code cells as needed.
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

EPOCHS = 100
BATCH_SIZE = 80

print('processed')

def LeNet(x):
    mu = 0
    sigma = 0.125
    
    # Layer 1: Convolution: input(32, 32, 1) output(28, 28, 6)
    W1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma),name="W1")
    x = tf.nn.conv2d(x, W1, strides=[1,1,1,1], padding='VALID')
    b1 = tf.Variable(tf.zeros(6), name="b1")
    x = tf.nn.bias_add(x,b1)
    print("layer 1:",x.get_shape())
    
    # Layer 1: Activation Function
    x = tf.nn.relu(x)
    
    # Layer 1: Pooling: input(28, 28, 6) output(14, 14, 6)
    x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    layer1 = x
    
    # Layer 2: Convolution: input(14, 14, 6) output(10, 10, 16)
    W2 = tf.Variable(tf.truncated_normal(shape=(5,5,6,16), mean=mu, stddev=sigma), name="W2")
    x = tf.nn.conv2d(x, W2, strides=[1,1,1,1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16), name="b2")
    x = tf.nn.bias_add(x,b2)
    print("layer 2 shape:",x.get_shape())

    # Layer 2: Activation
    x = tf.nn.relu(x)
    
    # Layer 2: Pooling: input(10, 10, 16) output(5, 5, 16)
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    layer2 = x
    
    # Layer 3: Convolution: input(5, 5, 16) output(1, 1, 400)
    W3 = tf.Variable(tf.truncated_normal(shape=(5,5,16,400), mean=mu, stddev=sigma), name="W3")
    x = tf.nn.conv2d(x, W3, strides=[1,1,1,1], padding='VALID')
    b3 = tf.Variable(tf.zeros(400), name="b3")
    x = tf.nn.bias_add(x,b3)
    print("layer 3 shape:",x.get_shape())

    # Layer 3: Activation
    x = tf.nn.relu(x)
    layer3 = x
    
    # Layer 3: Flatten: input(5,5,16) output(400)
    layer2flat = flatten(layer2)
    print("layer3flat shape:",layer2flat.get_shape())
    
    # Layer 3: Flatten X: input(1,1,400) output(400)
    xflat = flatten(x)
    print("xflat shape:",xflat.get_shape())
    
    # Layer 3: Concat: input(400 + 400) output(800)
    x = tf.concat([xflat, layer2flat], 1)
    print("x shape:",x.get_shape())
    
    # Layer 3: Dropout
    x = tf.nn.dropout(x, keep_prob)
    
    # Layer 4 (Fully Connected): input(800) output(43)
    W4 = tf.Variable(tf.truncated_normal(shape=(800, 43), mean = mu, stddev = sigma), name="W4")
    b4 = tf.Variable(tf.zeros(43), name="b4") 
    logits = tf.add(tf.matmul(x, W4), b4)
    print("layer 4 shape:",logits.get_shape())

    return logits

print('BINGO!')


# In[7]:


tf.reset_default_graph() 

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, n_classes)

print('done')


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[8]:


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
rate = 0.0009

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)

loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        batch_x = np.expand_dims(batch_x, axis=3)
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

print('done')


# In[9]:


from sklearn.utils import shuffle


saver = tf.train.Saver()
X_training_set = X_train_processed
X_validation_set = X_valid_processed

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_training_set)

    print('Training...\n')
    
    for i in range(EPOCHS):
        X_training_set, y_train = shuffle(X_training_set, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_training_set[offset:end], y_train[offset:end]
            batch_x = np.expand_dims(batch_x, axis=3) 
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.6})

        validation_accuracy = evaluate(X_validation_set, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, 'lenet')
    print('Model Saved')


# In[10]:


import tensorflow as tf

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'lenet')
    validation_accuracy = evaluate(X_valid_processed, y_valid)
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    
with tf.Session() as sess:
    saver.restore(sess, 'lenet')
    test_accuracy = evaluate(X_test_processed, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[11]:


### Load the images and plot them here.
### Feel free to use as many code cells as needed.

import os
from scipy.ndimage import imread


german_test_images_dir = './test_images_jpg/'
traffic_signs = os.listdir(german_test_images_dir)
X_images = []

for f in traffic_signs:
    if f.endswith('.jpg'):
        image = imread(german_test_images_dir + f)
        X_images.append(image)

num_images = len(X_images)


# In[12]:


import matplotlib.pyplot as plt

fig, axarr = plt.subplots(num_images)
for img_num, image in enumerate(X_images):
    axarr[img_num].imshow(image)
    axarr[img_num].axis('off')

plt.show()


# In[13]:


SPEED_LIMIT_80 = 5
SPEED_LIMIT_120 = 8
PRIORITY_ROAD = 12
NO_PASSING = 9
YIELD = 13
SPEED_LIMIT_60 = 3
NO_VEHICLES = 15
BEWARE_OF_ICE_SNOW = 30
STOP = 14
ROAD_WORK = 25


y_images = [
    SPEED_LIMIT_80,
    SPEED_LIMIT_120,
    PRIORITY_ROAD,
    NO_PASSING,
    YIELD,
    SPEED_LIMIT_60,
    NO_VEHICLES,
    BEWARE_OF_ICE_SNOW,
    STOP,
    ROAD_WORK
    
]

label_map = pd.read_csv('./signnames.csv')

X_gray_images = preprocess(X_images)

fig, axarr = plt.subplots(num_images)

for img_num, image in enumerate(X_gray_images):
    axarr[img_num].imshow(image, cmap='gray')
    axarr[img_num].axis('off')

plt.show()


# ### Predict the Sign Type for Each Image

# In[14]:


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'lenet')
    x_batch = np.expand_dims(X_gray_images, axis=3) # TF requires 4 dimensions
    result_logits = sess.run(logits, feed_dict={x: x_batch, y: y_images, keep_prob: 1.0})

    predictions = tf.argmax(result_logits, 1).eval()
    compare_prediction = tf.equal(predictions, y_images)
    accuracy = tf.reduce_mean(tf.cast(compare_prediction, tf.float32)).eval()

    for idx, prediction in enumerate(predictions):
        prediction_name = label_map['SignName'][prediction]
        true_label = label_map['SignName'][y_images[idx]]
        print("PREDICTION {}".format(idx+1))
        print("Prediction: {}\nTrue label: {}\n".format(prediction_name, true_label))


# ### Analyze Performance

# In[15]:


### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

correct_prediction = tf.equal(predictions, y_images)
print("The model is {:.2f}% accurate.".format(accuracy*100))


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[16]:


### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

from sklearn.preprocessing import normalize
with tf.Session() as sess:
    for idx, result_logit in enumerate(result_logits):
        plt.imshow(X_images[idx])
        plt.axis('off')
        plt.show()
        result_logit = normalize(result_logit.reshape(1, -1), norm='max', axis=1)[0]

        k = 5
        print("Top {} Predictions:".format(k))
        top_k_predictions = tf.nn.top_k(tf.nn.softmax(result_logit), k)

        top_k_predictions_indices = top_k_predictions[1].eval()
        top_k_predictions_probs = top_k_predictions[0].eval()

        for prediction_index, prediction_prob in zip(top_k_predictions_indices, top_k_predictions_probs):
            prediction_name = label_map['SignName'][prediction_index]
            print("'{}' with probability: {:.3}".format(prediction_name, prediction_prob))


# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[ ]:


### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

