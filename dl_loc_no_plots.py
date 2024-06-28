
# coding: utf-8


################################################################################################
# D. Milioris - IMAGE CLASSIFICATION FOR INDOOR ROOM-LEVEL LOCALIZATION BASED ON DEEP LEARNING #                                                                                     #                          
# Partial code from Kaggle competition and MIT project on indoor localization via images       #
################################################################################################

import time
import math
import random

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
import dataset
import cv2

from sklearn.metrics import confusion_matrix
from datetime import timedelta

#%matplotlib inline

print("Import Process OK")


# In[2]:


## CONFIGURATION OF THE CNNs.

# Convolutional Layer 1.
filter_size1 = 3 
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions - square images
img_size = 128 # from 128

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
# Corridor1, Corridor2, Corridor3, Coffee Area - Bell Labs, VX campus
classes = ['cor_1', 'cor_2', 'cor_3', 'coffee_area']
num_classes = len(classes)

# batch size
batch_size = 12 # Change from 32 to 12

# validation percentage
validation_size = .16

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping

train_path = 'data/train/'
test_path = 'data/test/'
checkpoint_dir = "models/"

print("Layers and paths set!")


# In[3]:


## DATA LOADING

data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images, test_ids = dataset.read_test_set(test_path, img_size)

print("Read data; training and test")


# In[4]:


## PRINT THE SIZE OF DATASETS TO VERIFY

print("Size of datasets:")
print("- Training-set:\t\t{}".format(len(data.train.labels))) #.format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(test_images)))
print("- Validation-set:\t{}".format(len(data.valid.labels))) #.format(len(data.valid.labels)))


# In[5]:


## Helper funtion: 9 images in 3x3 grid; or less

def plot_images(images, cls_true, cls_pred=None):
    
    if len(images) == 0:
        print("no images to show")
        return 
    else:
        random_indices = random.sample(range(len(images)), min(len(images), 9))
        
        
    images, cls_true  = zip(*[(images[i], cls_true[i]) for i in random_indices])
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_size, img_size, num_channels))

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# In[9]:


## Plot testing to see if images are correctly represented/loaded

# Get some random images and their labels from the train set.

images, cls_true  = data.train.images, data.train.cls

# Plot the images and labels using our helper-function above.
#plot_images(images=images, cls_true=cls_true)


# In[10]:


## INITIALIZING TENSORFLOW WITH RANDOM VARIABLES - DEFINED IN THE TENSOR FLOW GRAPH
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


# In[11]:


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# In[12]:


## NEW CONV LAYER
##
## ASSUMPTIONS FOR INPUT - 4 dim tensor: 
## - Image number
## - X-axis of each image
## - Y-axis of each image
## - Channels of each image
##
## OUTPUT - 4 dim tensor:
## Image number as input
## X-axis; if 2x2 pooling is used then the height and width of the input is divided by 2
## Channels produced by the CONV filters

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


# In[13]:


## A convolutional layer produces an output tensor with 4 dimensions. 
## We will add fully-connected layers after the convolution layers, so we need to 
## reduce the 4-dim tensor to 2-dim which can be used as input to the fully-connected layer.

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


# In[14]:


## Create a new fully-connected layer in the computational graph for TensorFlow. 
## It's not calculated here; just adding the mathematical formulas to the TensorFlow graph.
## INPUT: 2-dim tensor of shape [num_images, num_inputs]. 
## OUTPUT: 2-dim tensor of shape [num_images, num_outputs].

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# In[15]:


## Placeholder variables serve as the input to the TensorFlow computational graph. 
## We may change that each time we execute the graph.
## The data-type is set to float32. 
## None means that the tensor may hold an arbitrary number of images. 
## Each image is a vector of length img_size_flat.

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')


# In[16]:


## CONV layers expect x to be encoded as a 4-dim tensor so we have to reshape it so its shape is instead [num_images, img_height, img_width, num_channels]. 
## img_height == img_width == img_size and num_images can be inferred automatically by using -1 for the size of the first dimension.

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])


# In[17]:


## Placeholder variable for the true labels associated with the images that were input in the placeholder variable `x`.
## The shape of this placeholder variable is `[None, num_classes]` which means it may hold an arbitrary number of labels.
## Each label is a vector of length `num_classes`.

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')


# In[18]:


## TensorFlow operator. Placeholder variable for the class-number; calculation using argmax.

y_true_cls = tf.argmax(y_true, dimension=1)


# In[19]:


## First CONV layer
## INPUT: x_image
## Creates num_filters1 different filters, each having width and height equal to filter_size1
## Down-sample the image so it is half size by using 2x2 max-pooling

layer_conv1, weights_conv1 =     new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)


# In[20]:


## First CONV layer

layer_conv1


# In[21]:


## Second and third CONV layers
## INPUT: The output from the first and second CONV layer respectively
## The number of input channels corresponds to the number of filters in the previous convolutional layer

layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)


# In[22]:


## Second CONV layer

layer_conv2


# In[23]:


## Check documentation in the previous cells

layer_conv3, weights_conv3 =     new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True)


# In[24]:


## Third CONV layer

layer_conv3


# In[25]:


## The CONV layers output 4-dim tensors
## INPUT in a fully-connected network; requires for the tensors to be reshaped or flattened to 2-dim tensors

layer_flat, num_features = flatten_layer(layer_conv3)


# In[26]:


## Flatten layer

layer_flat


# In[27]:


## Num of features

num_features


# In[28]:


## Add a fully-connected layer to the network
## INPUT: flattened layer from the previous convolution 
## The number of neurons or nodes in the fully-connected layer is fc_size
## ReLU is used so we can learn non-linear relations.

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)


# In[29]:


## CHECKPOINT: The output of the fully-connected layer is a tensor with shape (?, 128) 
## The ? means there is an arbitrary number of images and fc_size == 128 

layer_fc1


# In[30]:


## Add another fully-connected layer that outputs vectors of length num_classes for determining which of the classes the input image belongs to
## Note that ReLU is not used in this layer

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)


# In[31]:


layer_fc2


# In[32]:


## Likelihood of the input image belongs to each one of the classes
## Normalize the numbers so that each element is limited between zero and one and all the elements sum to one
## Using the so-called softmax function

y_pred = tf.nn.softmax(layer_fc2)


# In[33]:


## The class-number is the index of the largest element.

y_pred_cls = tf.argmax(y_pred, dimension=1)


# In[34]:


## The function calculates the softmax internally so we must use the output of layer_fc2 directly rather than y_pred which has already had the softmax applied

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)


# In[35]:


## Take the average of the cross-entropy for all the image classifications

cost = tf.reduce_mean(cross_entropy)


# In[36]:


## Optimization method
## AdamOptimizer - advanced form of Gradient Descent
## Just adding the optimizer-object to the TensorFlow for later execution

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


# In[37]:


## Vector of booleans; the predicted class equals the true class of each image

correct_prediction = tf.equal(y_pred_cls, y_true_cls)


# In[38]:


## Classification accuracy by first type-casting the vector of booleans to floats
## False --> 0; True --> 1
## Calculate the average of these numbers

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[39]:


## Run TensorFlow
##

session = tf.Session()


# In[40]:


## Initialize variables for weights and biases before of their optimization

#session.run(tf.initialize_all_variables())
session.run(tf.global_variables_initializer())


# In[41]:


## I am using a small batch of images in each iteration of the optimizer
## Might lower that number down if it crashes --> increase the optimization iterations

train_batch_size = batch_size


# In[42]:


## Printing Training and Validation accuracy, along with Validation loss
##

def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


# In[43]:


## Performing a number of optimization iterations so as to gradually improve the variables of the network layers
## In each iteration a new batch of data is selected from the training-set
## TensorFlow executes the optimizer using those training samples 
## The progress is printed every 100 iterations.

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.

    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()
    
    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,
                   total_iterations + num_iterations):  
        
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]

        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        
        feed_dict_validate = {x: x_valid_batch,
                              y_true: y_valid_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
        

        # Print status at end of each epoch (defined as full pass through training dataset).
        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))
            
            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
            
            if early_stopping:    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

    # Update the total number of iterations performed.
    total_iterations += num_iterations
    #print 'Number of iterations: ', total_iterations
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))


# In[44]:


## Plotting examples of images from the testing set that have been misclassified
##

def plot_example_errors(cls_pred, correct):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.valid.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.valid.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


# In[45]:


## Plot the confusion matrix
##

def plot_confusion_matrix(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.valid.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# In[46]:


## PERFORMANCE EVALUATION
## Reuse the results by calling the previous functions directly from this function
## That way the classifications don't have to be recalculated by each function
## Might need to split the testing set into smaller baches because of crash on my poor computer!

def print_validation_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    #global x
    # Number of images in the test-set.
    num_test = len(data.valid.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.valid.images[i:j, :].reshape(batch_size, img_size_flat)
        

        # Get the associated labels.
        labels = data.valid.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    cls_true = np.array(data.valid.cls)
    cls_pred = np.array([classes[x] for x in cls_pred]) 

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


# In[47]:


## PERFORMANCE AFTER 1 OPTIMIZATION ITERATION
## 

optimize(num_iterations=1)
#print_validation_accuracy()


print("==== validation accuracy after %d iterations ====" % total_iterations)
#print_validation_accuracy(show_example_errors=True)
print_validation_accuracy()

# In[49]:


## PERFORMANCE AFTER 100 OPTIMIZATION ITERATIONS
## 

optimize(num_iterations=99)  # We already performed 1 iteration above.


print("==== validation accuracy after %d iterations ====" % total_iterations)
#print_validation_accuracy(show_example_errors=True)
print_validation_accuracy()

# In[50]:


## PERFORMANCE AFTER 1000 OPTIMIZATION ITERATIONS
##  

optimize(num_iterations=900)  # We performed 100 iterations above.


print("==== validation accuracy after %d iterations ====" % total_iterations)
#print_validation_accuracy(show_example_errors=True)
print_validation_accuracy()

##


## PERFORMANCE AFTER 10K OPTIMIZATION ITERATIONS
## 

#optimize(num_iterations=9000) # We performed 1000 iterations above.


# In[ ]:


#print_validation_accuracy(show_example_errors=True, show_confusion_matrix=True)


# In[ ]:


##
## TESTING SAMPLE IMAGE
##

#plt.axis('off')

test_coffee_area = cv2.imread('coffee_area_test120.jpg')
#plt.imshow(test_coffee_area)
test_coffee_area = cv2.resize(test_coffee_area, (img_size, img_size), cv2.INTER_LINEAR) / 255


# In[ ]:


##
## TESTING SAMPLE IMAGE
##

test_cor_1 = cv2.imread('cor_1_test091.jpg')
#plt.imshow(test_cor_1)
test_cor_1 = cv2.resize(test_cor_1, (img_size, img_size), cv2.INTER_LINEAR) / 255


# In[ ]:


##
## TESTING SAMPLE IMAGE
##

test_cor_2 = cv2.imread('cor_2_test122.jpg')
#plt.imshow(test_cor_2)
test_cor_2 = cv2.resize(test_cor_2, (img_size, img_size), cv2.INTER_LINEAR) / 255


# In[ ]:


##
## TESTING SAMPLE IMAGE
##

test_cor_3 = cv2.imread('cor_3_test178.jpg')
#plt.imshow(test_cor_3)
test_cor_3 = cv2.resize(test_cor_3, (img_size, img_size), cv2.INTER_LINEAR) / 255


# In[ ]:


##
## CHECK IF THE SAMPLE IMAGES ARE CLASSIFIED CORRECTLY
##

def sample_prediction(test_im):
    
    feed_dict_test = {
        x: test_im.reshape(1, img_size_flat),
        y_true: np.array([[0, 0, 0, 0]])
    }

    test_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    return classes[test_pred[0]]

print("Predicted class for test_cor_1: {}".format(sample_prediction(test_cor_1)))
print("Predicted class for test_cor_2: {}".format(sample_prediction(test_cor_2)))
print("Predicted class for test_cor_3: {}".format(sample_prediction(test_cor_3)))
print("Predicted class for test_coffee_area: {}".format(sample_prediction(test_coffee_area)))


# In[ ]:


## VISUALIZATION OF THE WEIGHTS OF THE CNN AND THE OUTPUT IMAGES
##

def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.
    
    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# In[ ]:


## PLOT THE OUTPUT OF A CONV LAYER
##

def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.
    
    image = image.reshape(img_size_flat)

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


##


## Plot the filter-weights for the first convolutional layer
## Note that positive weights are red and negative weights are blue

#plot_conv_weights(weights=weights_conv1)


# In[ ]:


## Applying each of these convolutional filters to the first input image gives the following output images, which are then used as input to the second convolutional layer
## Note that these images are down-sampled to about half the resolution of the original input image

#plot_conv_layer(layer=layer_conv1, image=image1)


# In[ ]:


## Applying the convolutional filters to the second image

#plot_conv_layer(layer=layer_conv1, image=image2)


# In[ ]:


## Plot the filter-weights for the second convolutional layer
## There are 16 output channels from the first conv-layer
## It means there are 16 input channels to the second conv-layer
## The second conv-layer has a set of filter-weights for each of its input channels
## We start by plotting the filter-weigths for the first channel

#plot_conv_weights(weights=weights_conv2, input_channel=0)


# In[ ]:


## There are 16 input channels to the second convolutional layer, so we can make another 15 plots of filter-weights like this
## We just make one more with the filter-weights for the second channel

#plot_conv_weights(weights=weights_conv2, input_channel=1)


# In[ ]:


## Applying these convolutional filters to the images that were ouput from the first conv-layer gives the following images
## Note that these are down-sampled yet again to half the resolution of the images from the first conv-layer

#plot_conv_layer(layer=layer_conv2, image=image1)


# In[ ]:


## Results of applying the filter-weights to the second image

#plot_conv_layer(layer=layer_conv2, image=image2)


# In[ ]:


## For every image in the /test directory it gives a prediction to the classes (probabilities). 
## Output: A .csv file which contains the probabilities of the classes. 

def write_predictions_for_classes(ims, ids):
     ims = ims.reshape(ims.shape[0], img_size_flat)
     preds = session.run(y_pred, feed_dict={x: ims})
     result = pd.DataFrame(preds, columns=classes)
     result.loc[:, 'id'] = pd.Series(ids, index=result.index)
     pred_file = 'predict_classes_VX.csv'
     result.to_csv(pred_file, index=False)


## Writes the predicted classes and their probabilities.
write_predictions_for_classes(test_images, test_ids)


## CLOSING TENSORFLOW

session.close()

