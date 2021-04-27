# Introduction
WYM - Wear Your Mask is a web application that utilizes a classifier to identify individuals not wearing a mask properly and crowdsourcing to gather data to train the classifier.
# WYM Dev Setup
- In the root directory, create an environment in env folder
    > python -m venv env
- Activate environment
    > env\Scripts\activate
- Install all required packages
    > python -m pip install -r requirements.txt
- Install pytorch
    - https://pytorch.org/
- Document all files in Directory section of docs/README.md and if the file is code, explain in detail in Code section of docs/README.md

# How To Contribute
Thank you for contributing to our project! Below, you will find instructions as to how to access our mturk sandbox HITs, as well as detailed instructions as to how to contribute successfully.
## Links
Bounding Box HIT: https://workersandbox.mturk.com/projects/3AG2YDBQJVF78HUSVYDQVDYIGN7XHG/tasks?ref=w_pl_prvw

Face Labeling HIT: https://workersandbox.mturk.com/projects/3ULSZRQKCIPVDH9RV3E38LN16R6UFI/tasks?ref=w_pl_prvw

Video Instruction: https://vimeo.com/541376394 
## Instructions
### Bounding Box
1. For this task, you will be presented with a picture containing one or more people who may or may not be wearing masks.

2. On the bottom of the task, the bounding box button should already be selected. If not, please select this button.

3. For each person in the picture, draw a bounding box around their face.
    - You do not need to cover the entire head, as long as the entire face is encapsulated in the box.
    - If a face is slightly covered by something or overlapped, you do not need to draw a box around it. Only draw boxes around unobstructed faces.
    - Make sure you capture every visible face, whether or not they are wearing a mask
4. After you have drawn a box around all unobstructed faces, submit and move on to the next task.
### Face Labeling
1. For this task, you will be presented with 9 pictures of zoomed in faces that may or may not be wearing a mask
2. For each face, open the dropdown below it and choose whether or not the face is wearing a mask properly, wearing a mask improperly, or not wearing a mask at all. 
    - The faces may not be completely visible or heavily pixelated, please answer to the best of your ability.
    - We constitute wearing a mask properly as having the mask cover the nose and mouth. We constitute wearing a mask improperly as having the mask not completely cover the nose and mouth.
3. After you have labeled all 9 faces, please move on to the next task.
# Documentation
Refer to README.md in docs directory for documentation on the files in various directories
Refer to README.md in src directory for information on how to run code
