# -*- coding: utf-8 -*-
import pandas as pd
from collections import defaultdict

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
    quality = defaultdict(lambda: [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    keys = ['Answer.wmc_qc', 'Answer.wmi_qc', 'Answer.nwm_qc']
    for _, row in df.iterrows():
        worker = row['WorkerId']
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
            # Compute total accuracy
            round((m[0][0] + m[1][1] + m[2][2]) / sum([sum(l) for l in m]), 3),
            # Check if worker is accurate for each classification label
            all([l[i] / sum(l) > 0.9 for i, l in enumerate(m)])
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
    votes = dict()
    if (iter_num >= 0):   
        for _ in range(iter_num):
            votes, worker_qual = em_iteration(rows, worker_qual)
    else:
        prev_worker_qual = dict()
        while prev_worker_qual != worker_qual:
            prev_worker_qual = worker_qual
            votes, worker_qual = em_iteration(rows, worker_qual)
    return sorted([
        # Get label corresponding to index of max weight
        (k, inverse_label_map[max(range(len(v)), key=v.__getitem__)])
        for k, v in votes.items()
    ])


def main():
    # Read in CVS result file with pandas
    # PLEASE DO NOT CHANGE
    result_df = pd.read_csv('sample_data/sample_hit_results.csv')

    # Call functions and output required CSV files
    quality, cm = worker_quality(result_df)
    df = pd.DataFrame(quality, columns=['WorkerId', 'Accuracy', 'GoodWorker'])
    df.to_csv('sample_data/sample_qc_output.csv', index=False)
    
    labels = em_vote(result_df, cm, -1)
    df = pd.DataFrame(labels, columns=['ImageUrl', 'Label'])
    df.to_csv('sample_data/sample_agg_output.csv', index=False)
    
    
    

if __name__ == '__main__':
    main()
