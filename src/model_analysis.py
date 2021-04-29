from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from os.path import join
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

label_map = {
    "Wearing Mask Correctly": 0,
    "Wearing Mask Incorrectly": 1,
    "Not Wearing Mask": 2
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
# This is a preliminary analysis so we will condsider failure to detect face a false classification
def calculate_preliminary_model_accuracy(labels):
    # Read model labels
    df = pd.read_csv(join(data_dir, analysis_dir, 'model_image_labels.csv'))
    model_labels = {k: v for (k, v) in df[['Image', 'Label']].values.tolist()}
    quality = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for image in labels:
        # Update confusion matrix
        if image in model_labels:
            quality[label_map[labels[image]]][label_map[model_labels[image]]] += 1
        else:
            num = label_map[labels[image]]
            quality[num][(num + 1) % len(label_map)] += 1

    return quality

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
    random_baseline = 1 / len(label_map)
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
    preliminary_scatter_plot(worker_qual, model_qual)

if __name__ == '__main__':
    main()