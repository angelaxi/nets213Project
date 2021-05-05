# -*- coding: utf-8 -*-
import pandas as pd
from collections import defaultdict
from os import mkdir
from os.path import join, isdir
from json import loads, dumps
from pytz import timezone
from dateutil.parser import parse

label_map = {
    "Wearing Mask Correctly": 0,
    "Wearing Mask Incorrectly": 1,
    "Not Wearing Mask": 2
}

inverse_label_map = {
    0: "Wearing Mask Correctly",
    1: "Wearing Mask Incorrectly",
    2: "Not Wearing Mask"
}

# Quality Control
def worker_quality(df):
    tz = { 'PDT': timezone('US/Pacific')}
    quality = defaultdict(lambda: [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    time = defaultdict(lambda: [0, 0])
    keys = ['Answer.wmc_qc', 'Answer.wmi_qc', 'Answer.nwm_qc']
    for _, row in df.iterrows():
        worker = row['WorkerId']
        # Compute time spend on task
        accept_time = parse(row['AcceptTime'], tzinfos=tz)
        submit_time = parse(row['SubmitTime'], tzinfos=tz)
        time[worker][0] += (submit_time - accept_time).seconds
        time[worker][1] += 1
        # Iterate over quality control values
        for i, key in enumerate(keys):
            label = row[key]
            # Check for invalid label
            if label in label_map:
                # Increment confusion matrix
                quality[worker][i][label_map[label]] += 1
            else:
                raise Exception("Unexpected label %s" % label)
            
    return sorted([
        (
            k, 
            time[k][1],
            time[k][0] / time[k][1],
            # Compute accuracies
            m[0][0] / sum(m[0]),
            m[1][1] / sum(m[1]),
            m[2][2] / sum(m[2]),
            (m[0][0] + m[1][1] + m[2][2]) / sum([sum(l) for l in m]),
            # Check if worker is accurate for each classification label
            all([l[i] / sum(l) >= 0.9 for i, l in enumerate(m)])
        ) for k, m in quality.items()]), { # Normalize Confusion Matrix
            k: [[0 if sum(m[0]) == 0 else m[0][0] / sum(m[0]), 
                 0 if sum(m[0]) == 0 else m[0][1] / sum(m[0]),
                 0 if sum(m[0]) == 0 else m[0][2] / sum(m[0])],
                [0 if sum(m[1]) == 0 else m[1][0] / sum(m[1]), 
                 0 if sum(m[1]) == 0 else m[1][1] / sum(m[1]),
                 0 if sum(m[1]) == 0 else m[1][2] / sum(m[1])],
                [0 if sum(m[2]) == 0 else m[2][0] / sum(m[2]), 
                 0 if sum(m[2]) == 0 else m[2][1] / sum(m[2]),
                 0 if sum(m[2]) == 0 else m[2][2] / sum(m[2])]]
            for k, m in quality.items()}

def em_worker_quality(rows, labels):
    quality = defaultdict(lambda: [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for _, row in rows.iterrows():
        worker = row['WorkerId']
        for i in range(1, 7):
            key = 'image' + str(i)
            label = row["Answer." + key]
            
            # Check for invalid label
            if label in label_map:
                label = label_map[label]
            else:
                raise Exception("Unexpected label %s" % label)
                
            weights = labels[row['Input.' + key]]
            
            # Get index of max element
            truelabel = max(range(len(weights)), key=weights.__getitem__)
            
            # Increment confusion matrix
            quality[worker][truelabel][label] += 1
            
    # Return normalized confusion matrix
    return { k: [[0 if sum(m[0]) == 0 else m[0][0] / sum(m[0]), 
                  0 if sum(m[0]) == 0 else m[0][1] / sum(m[0]),
                  0 if sum(m[0]) == 0 else m[0][2] / sum(m[0])],
                 [0 if sum(m[1]) == 0 else m[1][0] / sum(m[1]), 
                  0 if sum(m[1]) == 0 else m[1][1] / sum(m[1]),
                  0 if sum(m[1]) == 0 else m[1][2] / sum(m[1])],
                 [0 if sum(m[2]) == 0 else m[2][0] / sum(m[2]), 
                  0 if sum(m[2]) == 0 else m[2][1] / sum(m[2]),
                  0 if sum(m[2]) == 0 else m[2][2] / sum(m[2])]]
            for k, m in quality.items()}

# Aggregation
def em_votes(rows, worker_qual):
    votes = defaultdict(lambda: [0, 0, 0])
    for _, row in rows.iterrows():
        worker = row['WorkerId']
        worker_matrix = worker_qual[worker]
        for i in range(1, 7):
            key = 'image' + str(i)
            label = row["Answer." + key]
            
            # Check for invalid label
            if label in label_map:
                label = label_map[label]
            else:
                raise Exception("Unexpected label %s" % label)
                
            image = row['Input.' + key]
            
            # Increment weights for respective labels
            votes[image][0] += worker_matrix[0][label]
            votes[image][1] += worker_matrix[1][label]
            votes[image][2] += worker_matrix[2][label]
    return votes

def em_iteration(rows, worker_qual):
    votes = em_votes(rows, worker_qual)
    worker_qual = em_worker_quality(rows, votes)
    return votes, worker_qual

def em_vote(rows, worker_qual, iter_num):
    # Set worker quality to perfect if worker_qual is not a dictionary
    if not isinstance(worker_qual, dict):
        worker_qual = defaultdict(lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    votes = dict()
    # Iterate for iter_num
    if (iter_num >= 0):   
        for _ in range(iter_num):
            votes, worker_qual = em_iteration(rows, worker_qual)
    # Iterate until convergence
    else:
        prev_worker_qual = None
        while prev_worker_qual != worker_qual:
            prev_worker_qual = worker_qual
            votes, worker_qual = em_iteration(rows, worker_qual)
    return sorted([
        # Get label corresponding to index of max weight
        (k, inverse_label_map[max(range(len(v)), key=v.__getitem__)])
        for k, v in votes.items()
    ])

def em_vote(rows, worker_qual, iter_num):
    # Set worker quality to perfect if worker_qual is not a dictionary
    if not isinstance(worker_qual, dict):
        worker_qual = defaultdict(lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    votes = dict()
    # Iterate for iter_num
    if (iter_num >= 0):   
        for _ in range(iter_num):
            votes, worker_qual = em_iteration(rows, worker_qual)
    # Iterate until convergence
    else:
        prev_worker_qual = None
        while prev_worker_qual != worker_qual:
            prev_worker_qual = worker_qual
            votes, worker_qual = em_iteration(rows, worker_qual)
    return sorted([
        # Get label corresponding to index of max weight
        (k, inverse_label_map[max(range(len(v)), key=v.__getitem__)])
        for k, v in votes.items()
    ])

# Create CSV file input to train classification_model
def create_classification_model_input():
    # Read gold standard labels and generated image labels
    data_dir = '../data'
    s3 = 'https://wym-mask-images.s3.amazonaws.com/'
    labels = pd.read_csv(join(data_dir, 'analysis', 'image_labels.csv')).values.tolist()

    # Read bounding hit output to get bounding boxes
    df = pd.read_csv(join(data_dir, 'bounding_hit_output.csv'))
    df = df[['Input.image_url', 'Answer.annotatedResult.boundingBoxes']]
    df['Input.image_url'] = df['Input.image_url'].apply(lambda x: x.replace(s3, ''))
    df.columns = ['Image', 'BoundingBoxes']
    df = df.sort_values(['Image'])
    # Iterate over bounding boxes
    index = 0
    all_box_labels = []
    for _, row in df.iterrows():
        image = row['Image']
        bboxes = loads(row['BoundingBoxes'])
        box_labels = []
        sorted_str = sorted([str(j) for j in range(len(bboxes))])
        for i in sorted_str:
            # Generate crop image file name
            crop_image = "%s-%s.png" % (image[:len(image)-4], i)
            # Check that generated crop image file name matches actual file name
            if crop_image != labels[index][0]:
                raise Exception("Expected %s but was %s" % (crop_image, labels[index][0]))
            box_labels.append(labels[index][1])
            index += 1
        # Convert box_labels from string sorted order to integer sorted order
        box_labels = [box_labels[i] for i in sorted(range(len(sorted_str)), key=lambda i: int(sorted_str[i]))]
        # Append worker labels
        all_box_labels.append(dumps(box_labels))
    # Generate new csv with labels column
    df['Labels'] = all_box_labels
    df.to_csv(join(data_dir, 'classifier_input.csv'), index=False)
  

def main():
    data_dir = '../data/'
    analysis_dir = 'analysis'
    # Create analysis directory
    if not isdir(join(data_dir, analysis_dir)):
        mkdir(join(data_dir, analysis_dir))
    # Read in CSV result file with pandas
    result_df = pd.read_csv(join(data_dir, 'classification_hit_output.csv'))
    # Compute worker quality and confusion matrix from gold standard labels
    quality, cm = worker_quality(result_df)
    df = pd.DataFrame(quality, columns=[
        'WorkerId', 'TasksCompleted', 'TimePerTask', 'WearingMaskCorrectlyAccuracy',
        'WearingMaskIncorrectlyAccuracy', 'NotWearingMaskAccuracy', 'TotalAccuracy', 'GoodWorker'
    ])
    df.to_csv(join(data_dir, analysis_dir, 'gold_standard_quality.csv'), index=False)
    
    # 1 iteration EM with gold standard label performance as initial quality
    unconverged_weighted_labels = em_vote(result_df, cm, 1)

    # Append worker labels with gold standard labels
    s3 = 'https://wym-mask-images.s3.amazonaws.com/crop/'
    labels = unconverged_weighted_labels + pd.read_csv(join(data_dir, 'gold_standard_image_labels.csv')).values.tolist()
    labels = sorted([(url.replace(s3, ''), label) for (url, label) in labels])
    df = pd.DataFrame(labels, columns=['Image', 'Label'])
    df.to_csv(join(data_dir, analysis_dir, 'image_labels.csv'), index=False)
    
    create_classification_model_input()
    
    
    

if __name__ == '__main__':
    main()
