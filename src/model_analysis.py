from matplotlib.colors import LinearSegmentedColormap
from os.path import join
from collections import defaultdict
from json import loads
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

label_map = {
    "Wearing Mask Correctly": 0,
    "Wearing Mask Incorrectly": 1,
    "Not Wearing Mask": 2,
    "Duplicate": 3,
    "Unverifiable": 4,
    "Incorrect": 5
}
data_dir = '../data'
analysis_dir = 'analysis'

# Calculate worker accuracy on classification images
def calculate_worker_quality(labels):
    df = pd.read_csv(join(data_dir, 'classification_hit_output.csv'))
    s3 = 'https://wym-mask-images.s3.amazonaws.com/crop/'
    worker_qual = defaultdict(lambda: [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    keys = ['Answer.wmc_qc', 'Answer.wmi_qc', 'Answer.nwm_qc']
    for _, row in df.iterrows():
        worker = row['WorkerId']
        # Iterate over input images
        for i in range(1, 7):
            key = 'image' + str(i)
            label = row["Answer." + key]
            # Check for invalid label
            if label in label_map:
                image = row['Input.' + key].replace(s3, '')
                # Update confusion matrix
                worker_qual[worker][label_map[labels[image]]][label_map[label]] += 1
            else:
                raise Exception("Unexpected label %s" % label)
        # Iterate gold standard images
        for i, key in enumerate(keys):
            label = row[key]
            # Check for invalid label
            if label in label_map:
                # Update confusion matrix
                worker_qual[worker][i][label_map[label]] += 1
            else:
                raise Exception("Unexpected label %s" % label)
    return worker_qual

# Calculate model accuracy on segmentation and classification
# This is a preliminary analysis so we will consider failure to detect face a false classification
def calculate_preliminary_model_accuracy(labels):
    # Read model labels
    df = pd.read_csv(join(data_dir, analysis_dir, 'model_image_labels.csv'))
    model_labels = {k: v for (k, v) in df[['Image', 'Label']].values.tolist()}
    quality = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for image in labels:
        # Update confusion matrix
        num = label_map[labels[image]]
        if image in model_labels:
            quality[num][label_map[model_labels[image]]] += 1
        else:
            quality[num][(num + 1) % 3] += 1

    return quality

# Calculate model accuracy on classification
def calculate_model_accuracy():
    # Read model performance on validation
    df = pd.read_csv(join(data_dir, analysis_dir, 'model_performance.csv'))
    # Confusion matrix for 6 bounding box categories and 3 face categories
    quality = [[0, 0, 0] for _ in range(6)]
    # Stores average score for each bounding box category
    avg_scores = [[0, 0] for _ in range(6)]
    # Stores percentage of faces detected in an image
    detected_faces = [0, 0, 0, 0, 0]
    # Iterate over predictions for each validation image
    for _, row in df.iterrows():
        # Read columns
        scores = loads(row['PredictedScores'])
        labels = loads(row['PredictedLabels'])
        true_labels = loads(row['Labels'])
        num_undetected = row['UndetectedFaces']
        detected_faces[4] += num_undetected
        # Iterate over bounding box labels
        for i, label in enumerate(labels):
            label = label_map[label]
            true_label = label_map[true_labels[i]]
            # Increment confusion matrix
            quality[true_label][label] += 1
            # Update scores
            avg_scores[true_label][0] += scores[i]
            avg_scores[true_label][1] += 1
            # Update detection rate
            if true_label < 3:
                detected_faces[0] += 1
            else:
                detected_faces[true_label - 2] += 1

    return quality, [total / num for total, num in avg_scores], detected_faces

# Compute accuracy statistics for workers and model
def accuracy_bar_graph(worker_qual, model_qual, val_model_qual):
    # Aggregate worker results
    total_worker_qual = sum(np.array([cm for cm in worker_qual.values()]))
    # Compute worker accuracies
    worker_total = sum([sum(l) for l in total_worker_qual])
    worker_accs = [l[i] / sum(l) for (i, l) in enumerate(total_worker_qual)]
    worker_correct = total_worker_qual[0][0] + total_worker_qual[1][1] + total_worker_qual[2][2]
    worker_accs.append(worker_correct / worker_total)
    # Compute model accuracies
    model_total = sum([sum(l) for l in model_qual])
    model_accs = [l[i] / sum(l) for (i, l) in enumerate(model_qual)]
    model_correct = model_qual[0][0] + model_qual[1][1] + model_qual[2][2]
    model_accs.append(model_correct / model_total)
    # Compute validation model accuracies
    val_model_total = sum([sum(l) for l in val_model_qual])
    val_model_accs = [l[i] / sum(l) for (i, l) in enumerate(val_model_qual)]
    val_model_correct = val_model_qual[0][0] + val_model_qual[1][1] + val_model_qual[2][2]
    val_model_accs.append(val_model_correct / val_model_total)

    # Plot graph
    plt.figure(figsize=(10, 6))
    labels = ['WMC', 'WMI', 'NWM', 'Total']
    x = np.arange(len(labels)) * 2
    width = 0.4
    plt.bar(x - width, worker_accs, width=width, label='Worker')
    plt.bar(x, model_accs, width=width, label='Model Training')
    plt.bar(x + width, val_model_accs, width=width, label='Model Validation')
    plt.legend(bbox_to_anchor=(1.27, 0.5), loc='center right')
    plt.xticks(x, labels)
    plt.title("Accuracy of Workers and FasterRCNN model")
    plt.xlabel("Mask Categories")
    plt.ylabel("Percentage Accuracy")
    plt.subplots_adjust(right=0.8)
    plt.savefig(join(data_dir, analysis_dir, 'worker_model_accuracies.png'))
    plt.show()

# Plot probability of each label given the bounding box label in a bar graph
def probability_bar_graph(cm):
    cm = [[x / sum(l) for x in l] for l in cm]
    labels = ['WMC', 'WMI', 'NWM', 'Duplicate', 'Unverifiable', 'Incorrect']
    x = np.arange(len(labels)) * 5
    width = 2
    plt.bar(x, scores, width=width, color=['green', 'yellow', 'red', 'blue', 'orange', 'purple'])
    plt.title('Probability distribution of mask categories given bounding box categories')
    plt.xticks(x, labels)
    plt.xlabel('Bounding Box Categories')
    plt.ylabel('Percentage Confidence')
    plt.ylim((0, 1))
    plt.savefig(join(data_dir, analysis_dir, 'model_scores.png'))
    plt.show()

# Plot average scores for each bounding box category
def scores_bar_graph(scores):
    labels = ['WMC', 'WMI', 'NWM', 'Duplicate', 'Unverifiable', 'Incorrect']
    x = np.arange(len(labels)) * 5
    width = 2
    plt.bar(x, scores, width=width, color=['green', 'yellow', 'red', 'blue', 'orange', 'purple'])
    plt.title('Average model confidence score for each bounding box category')
    plt.xticks(x, labels)
    plt.xlabel('Bounding Box Categories')
    plt.ylabel('Percentage Confidence')
    plt.ylim((0, 1))
    plt.savefig(join(data_dir, analysis_dir, 'model_scores.png'))
    plt.show()

def detection_rate_pie_chart(detect_rate):
    labels = ['Detected', 'Undetected']
    sizes = detect_rate[::4]
    explode = (0, 0.1)
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=0, colors=['green', 'red'])
    plt.title('Percentage of faces detected by model')
    plt.savefig(join(data_dir, analysis_dir, 'face_detect.png'))
    plt.show()
    labels = ['Verifiable', 'Duplicate', 'Unverifiable', 'Incorrect']
    sizes = detect_rate[:4]
    explode = (0.1, 0.05, 0.05, 0.05)
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=10, colors=['green', 'blue', 'orange', 'purple'])
    plt.title('Percentage makeup of model bounding boxes')
    plt.savefig(join(data_dir, analysis_dir, 'bounding_boxe_categories.png'))
    plt.show()

# Draw scatter plot of workers and model with random and majority baseline
# Plot Image labeled vs Accuracy
# Color map that highlights minimum accuracy of each worker across all categories
def preliminary_scatter_plot(worker_qual, model_qual):
    xs = []
    ys = []
    cs = []
    max_total = 0
    for worker in worker_qual:
        m = worker_qual[worker]
        # Get total number of images labeled
        total = sum([sum(l) for l in m])
        # Save max number of images labeled
        if total > max_total:
            max_total = total
        # Get correct number of images labeled
        correct = m[0][0] + m[1][1] + m[2][2]
        # Append total and accuracy
        xs.append(total)
        ys.append(correct / total)
        # Find least accurate category
        min_acc = min([l[i] / sum(l) for i, l in enumerate(m)])
        cs.append(min_acc)
    # Compute graph values for model
    m = model_qual
    total = sum([sum(l) for l in m])
    if total > max_total:
        max_total = total
    correct = m[0][0] + m[1][1] + m[2][2]
    # Compute baselines
    majority_baseline = max([sum(l) for l in m]) / total
    random_baseline = 1 / 3
    min_acc = min([l[i] / sum(l) for i, l in enumerate(m)])
    # Plot graph
    cmap = LinearSegmentedColormap.from_list("", ["red","yellow","green"])
    plt.scatter(xs, ys, c=cs, cmap=cmap, vmin=0, vmax=1)
    plt.scatter([total], [correct / total], s = 100, label="FasterRCNN", c=[min_acc], cmap=cmap, vmin=0, vmax=1)
    plt.annotate("FasterRCNN", (total - 2000, correct / total + 0.015))
    plt.plot([0, max_total + 1000], [majority_baseline, majority_baseline], color=cmap(0), label="Majority Baseline")
    plt.plot([0, max_total + 1000], [random_baseline, random_baseline], color=cmap(random_baseline), label="Random Baseline")
    plt.title("Images labeled vs Accuracy of Workers and FasterRCNN model")
    plt.xlabel("Images labeled")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.xlim((7, max_total + 1000))
    plt.ylim((0.3, 1.05))
    plt.legend()
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Minimum accuracy across all mask categories', rotation=270)
    plt.savefig(join(data_dir, analysis_dir, 'worker_model_quality.png'))
    plt.show()


def main():
    # Read true labels into dictionary
    df = pd.read_csv(join(data_dir, analysis_dir, 'image_labels.csv'))
    labels = {k: v for (k, v) in df[['Image', 'Label']].values.tolist()}
    worker_qual = calculate_worker_quality(labels)
    model_qual = calculate_preliminary_model_accuracy(labels)
    val_model_qual, scores, detect_rate = calculate_model_accuracy()
    accuracy_bar_graph(worker_qual, model_qual, val_model_qual[:3])
    probability_bar_graph(val_model_qual)
    scores_bar_graph(scores)
    detection_rate_pie_chart(detect_rate)
    scatter_plot(worker_qual, model_qual, val_model_qual)

if __name__ == '__main__':
    main()