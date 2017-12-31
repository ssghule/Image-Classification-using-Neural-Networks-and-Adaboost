# Image-Classification-using-Neural-Networks-and-Adaboost

These days, all modern digital
cameras include a sensor that detects which way the camera is being held when a photo is taken. This meta-
data is then included in the image file, so that image organization programs know the correct orientation —
i.e., which way is “up” in the image. But for photos scanned in from film or from older digital cameras,
rotating images to be in the correct orientation must typically be done by hand.
The task in this project is to create a classifier that decides the correct orientation of a given image. </br></br>

__Data:__ The dataset is of images from the Flickr photo sharing website.
The images were taken and uploaded by real users from across the world, making this a challenging task
on a very realistic dataset. We’ll simply treat the raw
images as numerical feature vectors, on which we can then apply standard machine learning techniques. In
particular, we’ll take an n × m × 3 color image (the third dimension is because color images are stored as
three separate planes – red, green, and blue), and append all of the rows together to produce a single vector
of size 1 × 3mn. Two ASCII text files, one for the training dataset and one for testing contain the feature vectors. The training dataset consists of about 10,000 images, while the test set contains about 1,000. To generate this file, each image has been
to a very tiny “micro-thumbnail” of 8 × 8 pixels, resulting in an 8 × 8 × 3 = 192 dimensional feature vector.
The text files have one row per image, where each row is formatted like: </br>
photo_id correct_orientation r11 g11 b11 r12 g12 b12 ... </br>
where:</br>
• photo id is a photo ID for the image. </br>
• correct orientation is 0, 90, 180, or 270. </br>
• r11 refers to the red pixel value at row 1 column 1, r12 refers to red pixel at row 1 column 2, etc.,
each in the range 0-255. </br></br>
To run the code for training:</br>

__./orient.py train train_file.txt model_file.txt [model]__ </br>
where [model] is one of _nearest, adaboost, or nnet_. This program uses the data in train file.txt to
produce a trained classifier of the specified type, and save the parameters in model file.txt. </br>

__./orient.py test test_file.txt model_file.txt [model]__ </br>

The program loads in the trained
parameters from model file.txt, runs each test example through the model, displays the classification accuracy
(in terms of percentage of correctly-classified images), and outputs a file called output.txt which indicates the
estimated label for each image in the test file. The output file cooresponds to one test image per line,
with the photo id, a space, and then the estimated label. </br></br>

Here are more detailed specifications for each classifier.</br>
• nearest: At test time, for each image to be classified, the program finds the k “nearest” images
in the training file, i.e. the ones with the closest distance (least vector difference) in Euclidean space,
and have them vote on the correct orientation.</br>
• adaboost: Use very simple decision stumps that simply compare one entry in the image matrix to
another, e.g. compare the red pixel at position 1,1 to the green pixel value at position 3,8. </br>
• nnet: A fully-connected feed-forward network to classify image orientation and implements
the backpropagation algorithm to train the network using gradient descent.
