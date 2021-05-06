from os.path import join
from collections import defaultdict
from matplotlib.colors import LinearSegmentedColormap
from result_process import em_vote, worker_quality
import pandas as pd
import matplotlib.pyplot as plt

data_dir = '../data/'
analysis_dir = 'analysis'

label_map = {
    "Wearing Mask Correctly": 0,
    "Wearing Mask Incorrectly": 1,
    "Not Wearing Mask": 2
}

# Compute accuracies given df of true labels and predicted labels
def compute_accuracies(df, labels):
    s3 = 'https://wym-mask-images.s3.amazonaws.com/crop/'
    cm = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # Cross reference all labels
    for image, row in df.iterrows():
        true_label = label_map[row['Label']]
        pred_label = label_map[labels[join(s3, image)]]
        cm[true_label][pred_label] += 1
    return (
        cm[0][0] / sum(cm[0]),
        cm[1][1] / sum(cm[1]),
        cm[2][2] / sum(cm[2]),
        (cm[0][0] + cm[1][1] + cm[2][2]) / sum([sum(l) for l in cm])
    )

# Plot accuracy vs iteration given dataframes of predicted and true labels
# Start with initial quality specified by worker_qual
def accuracy_iteration_plot(result_df, true_df, worker_qual):
    init = worker_qual != None
    if not isinstance(worker_qual, dict):
        worker_qual = defaultdict(lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    prev_worker_qual = None
    xs = []
    wmcs = []
    wmis = []
    nwms = []
    totals = []
    i = 1
    # Iterate until convergence
    while prev_worker_qual != worker_qual:
        # Get labels and confusion matrix after one iteration
        labels, cm = em_vote(result_df, worker_qual, 1, return_dict=True)
        # Compute accuracies
        wmc_acc, wmi_acc, nwm_acc, total_acc = compute_accuracies(true_df, labels)
        # Update previous and current confusion matrices
        prev_worker_qual = worker_qual
        worker_qual = cm
        # Append data
        xs.append(i)
        wmcs.append(wmc_acc)
        wmis.append(wmi_acc)
        nwms.append(nwm_acc)
        totals.append(total_acc)
        i += 1
    plt.plot(xs, wmcs, color='green', label='WMC Accuracy')
    plt.plot(xs, wmis, color='yellow', label='WMI Accuracy')
    plt.plot(xs, nwms, color='red', label='NWM Accuracy')
    plt.plot(xs, totals, color='black', label='Total Accuracy')
    plt.xticks(range(1, i))
    plt.legend()
    plt.xlabel("Iteration number")
    plt.ylabel("Percentage Accuracy")
    if init:
        plt.title("Number of iterations vs aggregation accuracy with initial worker quality")
        plt.savefig(join(data_dir, analysis_dir, 'accuracy_iteration_with_initial.png'))
    else:
        plt.title("Number of iterations vs aggregation accuracy without initial worker quality")
        plt.savefig(join(data_dir, analysis_dir, 'accuracy_iteration_without_initial.png'))
    plt.show()

# Plot individual worker_accuracy vs iteration given dataframes of predicted and true labels
# Start with initial quality specified by worker_qual
def worker_accuracy_iteration_plot(result_df, true_df, worker_qual):
    init = worker_qual != None
    if not isinstance(worker_qual, dict):
        worker_qual = defaultdict(lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    prev_worker_qual = None
    xs = []
    ys = []
    cs = []
    i = 0
    s3 = 'https://wym-mask-images.s3.amazonaws.com/crop/'
    # Get workers that contributed to images with true labels
    workers = {
        row['WorkerId'] for (i, row) in result_df.iterrows() 
        if row['Input.image1':'Input.image6'].apply(
            lambda x: x.replace(s3, '')).isin(true_df.index).any()
    }

    # Iterate until convergence
    while prev_worker_qual != worker_qual:
        # Get confusion matrix after one iteration
        _, cm = em_vote(result_df, worker_qual, 1, return_dict=True)
        # Append data
        for worker in workers:
            m = cm[worker]
            xs.append(i)
            # Compute total accuracy
            acc = (m[0][0] + m[1][1] + m[2][2]) / sum([sum(l) for l in m])
            ys.append(acc)
            # Find least accurate category
            min_acc = min([l[i] / sum(l) if sum(l) > 0 else 0 for i, l in enumerate(m)])
            cs.append(min_acc)
        i += 1
        # Update previous and current confusion matrices
        prev_worker_qual = worker_qual
        worker_qual = cm
     # Plot graph
    cmap = LinearSegmentedColormap.from_list("", ["red","yellow","green"])
    plt.scatter(xs, ys, c=cs, cmap=cmap, vmin=0, vmax=1)
    plt.xlabel("Iteration Number")
    plt.ylabel("Percentage Accuracy")
    plt.xticks(range(i))
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Minimum accuracy across all mask categories', rotation=270)
    if init:
        plt.title("Number of iterations vs worker accuracy with initial worker quality")
        plt.savefig(join(data_dir, analysis_dir, 'worker_accuracy_iteration_with_initial.png'))
    else:
        plt.title("Number of iterations vs worker accuracy without initial worker quality")
        plt.savefig(join(data_dir, analysis_dir, 'worker_accuracy_iteration_without_initial.png'))
    plt.show()
    

def main():
    # Read in CSV result file with pandas
    result_df = pd.read_csv(join(data_dir, 'classification_hit_output.csv'))
    # Get true labels
    df = pd.read_csv(join(data_dir, 'true_labels.csv'))
    df = df.set_index(['Image'])
    # Compute worker quality and confusion matrix from gold standard labels
    _, cm = worker_quality(result_df)
    # 1 iteration EM with gold standard label performance as initial quality
    accuracy_iteration_plot(result_df, df, cm)
    accuracy_iteration_plot(result_df, df, None)
    worker_accuracy_iteration_plot(result_df, df, cm)
    worker_accuracy_iteration_plot(result_df, df, None)

if __name__ == '__main__':
    main()
