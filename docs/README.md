# Introduction
Our project, WYM - Wear Your Mask, is a live application that utilizes camera recording to monitor the surrounding and determine if any individuals are maskless or not wearing their masks properly.
# Directories
## data
- bounding_images: directory containing all input images for bounding box HITs
- classification_images: directory containing all input images for classification HITs
- analysis: directory containing analysis files after performance quality control and aggregation
    - gold_standard_quality.csv: quality control output with accuracy and good worker label for each worker
    - image_labels.csv: aggregation output from running EM with gold standard performance as initial quality for 1 iteration
- bounding_hit_input(0-1).csv: generated CSV input for bounding box HITs
- bounding_hit_output.csv: output from bounding box HITs
- bounding_images.txt: links to bounding box input images on S3 bucket
- classification_hit_input(0-1).csv: generated CSV input for classification HITs
- classification_images.txt: links to classification input images on S3 bucket
- classifier_input.csv: CSV input to train classifier with columns as bounding box image file names, bounding boxes, and labels corresponding to each bounding box
- gold_standard_image_labels.csv: manually assigned labels to 50 images of each of our 3 categories
- unlabeled_images.csv: list of image urls that have not been manually labeled
## docs
- flowchart.png: flowchart with workflow and components
- workflow.png: flowchart only listing workflow
- screenshots.pdf: Screenshot of interfaces for HITs and users
## src
Contains code for the project as well as sample inputs and outputs to our code

Refer to the Code section for more information about each file and any future improvements
# Components
## Collect images (1, Completed)
- Find images and datasets online of people wearing masks, without masks, or wearing masks improperly
- Store in S3 bucket
## Bounding Box Task (1, Completed)
- Provide images to MTurk workers to draw bounding boxes around faces of visible people
## Crop Images (2, Completed)
- Using the bounding box data collected from workers, resize and crop images to fit the bounding box size in order to get closer pictures of faces
- Write python code to automatically crop images based on bounding box data
- Approve bounding box if all cropped images contain entire face and clear
## Label Task (2, Completed) 
- Create MTurk HIT for workers to label the newly cropped pictures as Wearing Mask Correctly, Wearing Mask Incorrectly, or Not Wearing Mask.
- Design html template to allow workers to label multiple images at once, including quality control images
## Quality Control (2, Completed)
- Use gold standard labels for certain images in order to conduct a quality check on the worker’s labels
- Have workers label multiple images at once with a couple prelabeled images mixed in
- Write python code to compute worker quality based on gold standard labels
## Aggregate Labels (2, Completed)
- Use EM in conjunction with gold standard labels performance as initial worker quality to generate true label for each image
- Write code to generate true labels using Expectation Maximization algorithm along with gold standard labels
## Train a Classifier (3)
- Using the labels determined by the workers, train a classifier to automatically determine whether or not a face in an image is Wearing Mask Correctly, Wearing Mask Incorrectly, or Not Wearing Mask.
## Analyze Accuracy (3)
- Split the data into train/test/validation sets and run the classifier to compute accuracy
- Fine tune parameters to increase accuracy
- Test model on random pictures taken to evaluate if model is good enough in a realtime setting
## Web Application (4)
- Allow users to turn on camera for live recording
- Feed camera recording into our classifier to determine if people in the recording are wearing masks properly
- Display classifier results back to the user
# Data
## Quality Control
### Input
Dataframe result from our Image Labeling HIT task with the following columns:
- WorkerId: Id of the worker who completed the task
- Input.image(1-6): Image url of images we want workers to classify
- Input.wmc_qc: Image url of wearing mask correctly quality control image
- Input.wmi_qc: Image url of wearing mask incorrectly quality control image
- Input.nwm_qc: Image url of not wearing mask quality control image
- Answer.image(1-6): Classification label of images
- Answer.wmc_qc: Classification of wearing mask correctly quality control image
- Answer.wmi_qc: Classification of wearing mask incorrectly quality control image
- Answer.nwm_qc: Classification of not wearing mask quality control image
### Output
List with the following columns:
- WorkerId: Id of the worker who completed tasks
- TasksCompleted: Number of tasks completed by worker
- Accuracy: Percentage accuracy on quality control images
- GoodWorker: True if and only if the worker has 90% percent accuracy for all image categories: Wearing Mask Correctly, Wearing Mask Incorrectly, and Not Wearing Mask

Gold Standard Label Confusion Matrix: A 3x3 confusion matrix for our 3 image categories
## Aggregation 
### Input
Same Dataframe as Quality Control input

Confusion Matrix from Quality Control output
### Output
List with the following columns:
- Image: Image file name
- Label: Image label of either Wearing Mask Correctly, Wearing Mask Incorrectly, or Not Wearing Mask.
# Code
- <b>result_process.py</b>: contains quality control and aggregation functions to process results
    - Quality Control: 
        - <i>worker_quality(df)</i>: Computes worker quality from gold standard label answers
            - df: Dataframe from HIT result
            - output
                - List of three tuples in the following order: WorkerId, Number of tasks completed, Gold standard label accuracy, Whether the worker is a good worker or not
                - Gold standard label confusion matrix
        - <i>em_worker_quality(rows, labels)</i>: Computes weighted worker quality
            - rows: Dataframe from HIT result
            - labels: Dictionary storing label for each image in the form of an array of length 3 where the value of each index represents the weight of the label corresponding to that index
            - output: Confusion matrix for each worker
    - Aggregation:
        - <i>em_votes(rows, worker_qual)</i>: Computes labels given worker quality
            - rows: Dataframe from HIT result
            - worker_quality: Dictionary storing confusion matrix for each worker
            - output: Dictionary storing label for each image in the form of an array of length 3 where the value of each index represents the weight of the label corresponding to that index
        - <i>em_iteration(rows, worker_qual)</i>: Completes one EM iteration
            - rows: Dataframe from HIT result
            - worker_quality: Dictionary storing confusion matrix for each worker
            - output: 
                - labels: Dictionary storing label for each image in the form of an array of length 3 where the value of each index represents the weight of the label corresponding to that index
                - new_worker_quality: Dictionary storing new confusion matrix for each worker
        - <i>em_vote(rows, worker_qual, iter_num)</i>: Compute labels after iter_num iterations of the EM algorithm with initial worker quality specified by worker_qual
            - rows: Dataframe from HIT result
            - worker_quality: Dictionary storing initial confusion matrix for each worker
            - iter_num: Number of iterations to perform EM algorithm or until convergence if iter_num is less than 0
            - output: Sorted list of image urls and their respective string labels
        - <i>create_classification_model_input()</i>: Generates CSV input to train classifier with columns as bounding box image file names, bounding boxes, and labels corresponding to each bounding box
- <b>hit_process.py</b>: contains functions to preprocess inputs for HITs and postprocess HIT outputs
    - <i>create_bounding_image_urls()</i>: Create text file with bounding box input image urls on S3 bucket
    - <i>create_bounding_hit_inputs()</i>: Create input CSVs for the bounding box HIT task
    - <i>crop_images()</i>: Crop bounding box images based on bounding box HIT output and save cropped images
    - <i>create_classification_image_urls()</i>: Create text file with classification input image urls on S3 bucket
    - <i>create_classification_hit_inputs():</i>: Create CSV file inputs for classification HIT given gold standard labels

## sample_data
Sample data to test quality control and aggregation modules
- sample_hit_result.csv: sample hit result that is input to both the quality control and aggregation module
- sample_qc_out.csv: sample quality control output from gold standard labels
- sample_agg_output.csv: sample EM aggregation output
## hit_templates
HTML code for HIT templates
- bounding_box_mockup.html: Template for face bounding box HIT task
- classification_mockup.html: Template for face classification HIT task
## Future Considerations
We have implemented the full version of our quality control and aggregation modules. We utilize gold standard labels to get an initial quality control check for our EM algorithm to start on. We then let our algorithm converge to receive our labels. That being said, for our final labels that we use to train our classifier, we will cross reference multiple versions of our algorithm and see if they match. For any inconsistencies, we will manually verify the classification label. We expect that there will be few inconsistencies since it should fairly obviously whether a person is wearing their mask correctly or not. In other words, we expect little variance in the data. The versions of our algorithm that we will cross-reference are as follows:
- Converged EM with gold standard label performance as initial quality
- Converged EM assuming all workers are initially perfect
- Majority vote: EM assuming all workers are initially perfect for 1 iteration
- EM with gold standard label performance as initial quality for 1 iteration
