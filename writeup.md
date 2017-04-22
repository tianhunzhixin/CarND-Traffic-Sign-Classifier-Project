#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./images/image1.png "Traffic Sign 1"
[image5]: ./images/image2.png "Traffic Sign 2"
[image6]: ./images/image3.png "Traffic Sign 3"
[image7]: ./images/image4.png "Traffic Sign 4"
[image8]: ./images/image5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the gray image can ignore the more additional infomation of the image.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the value of pixel is 0-255 and the shape is 32x32x3. The normalization operation is more efficiant one in image data handling. Because it only handles the samples to the range of 0-1.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x16     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				|
| Flatten					|		outputs 400										|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 1x1x400     									|
| RELU					|										|
| Flatten					|		outputs 400	|
| Concat					|		outputs 800	|
| Dropout					|		|
| Fully connected		| outputs 43     |
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an CNN architecture. The best value of epochs I tested is 60-80, so I choose the middle value, 70. And the best batch size is 100 after tested the different values. For getting the better prediction of image, I decided the value of learning rate to be 0.0009.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.962
* test set accuracy of 0.94

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen? Maybe I will choose the LeNet-5 because of its accuracy.
* What were some problems with the initial architecture? For training dataset it maybe can got a high accuracy but act not well for vaild dataset.
* How was the architecture adjusted and why was it adjusted? There are some methods for this, like 1.convert the RGB image to gray image; 2. normalized the image; 3. apply the new LeNet architecture from [pdf](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf); 4. adjust the learning-rate and batch-size for getting high performance
* Which parameters were tuned? How were they adjusted and why? The parameters that should be tuned is like epochs, size of batch, learning rate, dropout value.
* What are some of the important design choices and why were they chosen? The layer of CNN is important, we should adjust them for participate situation. Of course, the other method like dropout should be applied for big dataset and it will perform well for big dataset.

If a well known architecture was chosen:

* What architecture was chosen? Of course LeNet
* Why did you believe it would be relevant to the traffic sign application? For classifing model, it can perform well like human.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? For test dataset, we got the accuracy of 0.94. It is high for prediction of image.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 
![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)      		| Speed limit (50km/h)   | 
| Stop     			| Stop   |
| General caution					| General caution											|
| Road work	      		| Road work					 				|
| Turn left ahead			| Turn left ahead      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 94%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (50km/h) sign (probability of 0.99), and the image does contain a Speed limit (60km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9905         			| Speed limit (50km/h)   									| 
| 0.00937     				| Speed limit (30km/h) 										|
| 0.00007917					| Speed limit (80km/h)											|
| 0.00001708	      			| Speed limit (60km/h)					 				|
| 0.00001536				    | End of speed limit (80km/h)      							|


For the second image, the model is relatively sure that this is a stop sign (probability of 0.9999), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99999         			| Stop   									| 
| 0.00000168     				| Yield 										|
| 0.000000524					| Go straight or right											|
| 0.000000129	      			| General caution					 				|
| 0.00000001128				    | Turn left ahead      							|

For the third image, the model is relatively sure that this is a General caution sign (probability of 1.0), and the image does contain a General caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| General caution   									| 
| 4.59524619e-34       | Traffic signals 										|
| 2.46456912e-35					| Children crossing											|
| 3.55592758e-37	      			| Right-of-way at the next intersection					 				|
| 0.00000000e+00				    | Speed limit (20km/h)      							|

For the forth image, the model is relatively sure that this is a Road work sign (probability of 1.0), and the image does contain a Road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Road work   									| 
| 4.76174744e-09 			| Slippery road 										|
| 5.75416654e-14					| Right-of-way at the next intersection											|
| 1.30588244e-17	      			| Children crossing					 				|
| 1.03384070e-20				    | Priority road      							|

For the fifth image, the model is relatively sure that this is a Turn left ahead sign (probability of 1.0), and the image does contain a Turn left ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Turn left ahead   									| 
| 3.33363538e-15	| Ahead only 										|
| 8.05330021e-16					| Go straight or left											|
| 7.98900962e-17	      			| Traffic signals				 				|
| 1.04783536e-22				    | Keep right      							|


