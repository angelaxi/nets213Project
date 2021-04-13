# Introduction
Our project, WYM - Wear Your Mask, is a live application that utilizes camera recording to monitor the surrounding and determine if any individuals are maskless or not wearing their masks properly.
# Collect images (1)
* Find images and datasets online of people wearing masks, without masks, or wearing masks improperly
# Bounding Box Task (1)
* Provide images to MTurk workers to draw bounding boxes around faces of visible people
# Resize Images (2)
* Using the bounding box data collected from workers, resize and crop images to fit the bounding box size in order to get closer pictures of faces
* Write python code to automatically crop images based on bounding box data
* Approve bounding box if all cropped images contain entire face and clear
# Label Task (1) 
* Create MTurk HIT that has workers label the newly resized pictures as properly wearing a mask, improperly wearing a mask, or not wearing a mask.
# Quality Check (2)
* Use gold standard labels for certain images in order to conduct a quality check on the worker’s labels
* Have workers label multiple images at once with a couple prelabeled images mixed in
* Write python code to compute worker quality based on gold standard label
# Aggregate Labels (2)
* Use EM in conjunction with gold standard labels to generate true label for each image
* Write code to generate true labels using Expectation Maximization algorithm along with gold standard labels
# Train a Classifier (3)
* Using the labels determined by the workers, train a classifier to automatically determine whether or not a face in an image is wearing a mask properly, wearing a mask improperly, or not wearing a mask.
# Analyze Accuracy (3)
* Split the data into train//test/validation sets and run the classifier to compute accuracy
* Fine tune parameters to increase accuracy
* Test model on random pictures taken to evaluate if model is good enough in a realtime setting
# Web Application (4)
* Allow users to turn on camera for live recording
* Feed camera recording into our classifier to determine if people in the recording are wearing masks properly
* Display classifier results back to the user
