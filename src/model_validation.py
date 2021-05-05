from os.path import join, isfile, isdir
from json import loads
import pandas as pd
import numpy as np
import cv2
import getopt
import sys

data_dir = '../data'
analysis_dir = 'analysis'

bounding_box_label_map = {
    1: "Wearing Mask Correctly",
    2: "Wearing Mask Incorrectly",
    3: "Not Wearing Mask",
    4: "Duplicate",
    5: "Unverifiable",
    6: "Incorrect",

}

color_map = {
    1: (0, 255, 0),
    2: (0, 255, 255),
    3: (0, 0, 255)
}

# Get user input, writes input on frame, and returns input after enter
def get_user_input(frame, x0, y0, dy, num_options):
    answer = ''
    while answer == '':
        temp = ''
        key = 0
        while True:
            key = cv2.waitKey(0)
            # Check for backspace
            if key >= 0x110000 or key < 0:
                answer = 'q'
                break
            elif key == 13:
                break
            elif key == 8:
                temp = temp[:-1]
            else:
                temp += chr(key)
            # Draw frame
            cv2.putText(frame, temp, (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.imshow('Image', frame)
            frame[y0 - dy - 5: y0 + 5, :] = (255,255,255)
        if temp == 'q':
            answer = temp
        elif temp.isnumeric():
            temp = int(temp)
            # Check if number if within range
            if temp >= 0 and (num_options < 0 or temp < num_options):
                answer = temp
            else:
                cv2.putText(frame, 'Input is not within range. Please try again', (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                cv2.imshow('Image', frame)
                frame[y0 - dy - 5: y0 + 5, :] = (255,255,255)
        else:
            cv2.putText(frame, 'Input is not numeric. Please try again.', (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.imshow('Image', frame)
            frame[y0 - dy - 5: y0 + 5, :] = (255,255,255)
    return answer

def main():
    # Get model prediction son validation images
    result_f = join(data_dir, analysis_dir, 'model_image_preds.csv')
    result_df = pd.read_csv(result_f)
    # Read model_performance.csv
    f = join(data_dir, analysis_dir, 'model_performance.csv')
    df = None
    if isfile(f):
        df = pd.read_csv(f)
        df = df.set_index(['Image'])
    else:
        df = pd.DataFrame(columns=[
            'Image', 'PredictedBoundingBoxes', 'PredictedLabels', 'PredictedScores', 
            'Labels', 'UndetectedFaces'
        ])
        df = df.set_index(['Image'])
        df.to_csv(f, index=True)
    
    # Parse validation image directory
    image_dir = join(data_dir, 'validation_images')
    opts, args = getopt.getopt(sys.argv[1:], "i:")
    for o, a in opts:
        if o == "-i":
            if isdir(a):
                image_dir = a
            else:
                raise Exception('%s is not a directory' % a)
    # Define user input strings
    verify_string = [
        "Evaluate the bounding box in color.",
        "Enter q to quit.",
        "0: I can correctly label the face in the bounding box",
        "1. The bounding box is a duplicate (There already exists",
        "a bounding box over the same face)",
        "2. The bounding box is unverifiable (The face in the bounding",
        "is too blurry or small to be labeled)",
        "3. The bounding box is incorrect (The bounding box does not",
        "capture a face)"
    ]
    label_string = [
        "Label the face in the bounding box in color disregarding the",
        "predicted label.",
        "Enter q to quit.",
        "0: Wearing Mask Correctly (The mask completely covers the nose",
        "and mouth)",
        "1. Wearing Mask Incorrectly (The mask does not completely cover",
        "either the nose or the mouth)",
        "2. Not Wearing Mask (There is no mask on the face)"
    ]
    undetected_string = [
        "How many more faces can you label that does not have a bounding box?",
        "Enter q to quit."
    ]
    # Define user input parameters
    user_enter_q = False
    # Define display parameters
    black = (0, 0, 0)
    # Iterate over predictions
    for _, row in result_df.iterrows():
        if user_enter_q:
            break
        image = row['Image']
        # Check if prediction has already been evaluated
        if image in df.index:
            continue
        # Resize image
        frame = cv2.imread(join(image_dir, image))
        width = frame.shape[1]
        height = frame.shape[0]
        scale = min(630 / height, 1120 / width)
        scale_width = int(scale * width)
        scale_height = int(scale * height)
        frame = cv2.resize(frame, (scale_width, scale_height))
        blank_image = np.zeros((scale_height + 300, scale_width,3), np.uint8)
        blank_image[:,:] = (255,255,255)
        blank_image[0:scale_height, 0:scale_width] = frame
        frame = blank_image
        # Define text parameters
        x0 = 10
        y0 = scale_height + 25
        dy = 20
        # Get image predictions
        bboxes = loads(row['BoundingBoxes'])
        labels = loads(row['Labels'])
        scores = loads(row['Scores'])
        true_labels = []
        # Evaluate each bounding box
        for i, bbox in enumerate(bboxes):
            currframe = frame.copy()
            # Get label
            class_num = labels[i]
            xmin, ymin, xmax, ymax = [int(scale * x) for x in bbox]
            color = color_map[class_num]
            label = bounding_box_label_map[class_num]
            # Draw computed bounding boxes
            cv2.putText(currframe, label, (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            cv2.rectangle(currframe, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), black, 2)
            # Get user input to determine if they can label bounding box
            answer_y0 = y0 + (len(verify_string) + 1) * dy
            for i, line in enumerate(verify_string):
                cv2.putText(currframe, line, (x0, y0 + i * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 1)
            cv2.imshow('Image', currframe)
            answer = get_user_input(currframe, x0, answer_y0, dy, 4)
            if answer == 'q':
                user_enter_q = True
                break
            # If user cannot label bounding box, continue
            if answer > 0 and answer < 4:
                true_labels.append(bounding_box_label_map[answer + 3])
                continue
            # Get user input to label bounding box
            currframe[scale_height:, :] = (255, 255, 255)
            answer_y0 = y0 + (len(label_string) + 1) * dy
            for i, line in enumerate(label_string):
                cv2.putText(currframe, line, (x0, y0 + i * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 1)
            cv2.imshow('Image', currframe)
            answer = get_user_input(currframe, x0, answer_y0, dy, 3)
            if answer == 'q':
                user_enter_q = True
                break
            true_labels.append(bounding_box_label_map[answer + 1])
        else:
            # Get user input to determine how many faces the model missed=
            answer_y0 = y0 + (len(undetected_string) + 1) * dy
            for i, line in enumerate(undetected_string):
                cv2.putText(frame, line, (x0, y0 + i * dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, black, 1)
            cv2.imshow('Image', frame)
            answer = get_user_input(frame, x0, answer_y0, dy, -1)
            if answer == 'q':
                break
            labels = [bounding_box_label_map[i] for i in labels]
            df.loc[image] = [bboxes, labels, scores, true_labels, answer]
    df.to_csv(f, index=True)        

    
    

if __name__ == '__main__':
    main()