from matplotlib.colors import LinearSegmentedColormap
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt

data_dir = '../data'
analysis_dir = 'analysis'

# Scatter Plot of Task Completed vs Accuracy
# Color map that highlights minimum accuracy of each worker across all categories
def worker_task_accuracy_scatter_plot(df):
    xs = []
    ys = []
    cs = []
    max_total = 0
    for _, row in df.iterrows():
        # Append total and accuracy
        total = row['TasksCompleted']
        if total > max_total:
            max_total = total
        xs.append(total)
        ys.append(row['TotalAccuracy'])
        # Find least accurate category
        min_acc = min([
            row['WearingMaskCorrectlyAccuracy'], 
            row['WearingMaskIncorrectlyAccuracy'],
            row['NotWearingMaskAccuracy']
        ])
        cs.append(min_acc)
    # Plot graph
    cmap = LinearSegmentedColormap.from_list("", ["red","yellow","green"])
    plt.scatter(xs, ys, c=cs, cmap=cmap, vmin=0, vmax=1)
    plt.title("Task completed vs Accuracy of Workers")
    plt.xlabel("Tasks Completed")
    plt.ylabel("Percentage Accuracy")
    plt.xscale("log")
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Minimum accuracy across all mask categories', rotation=270)
    plt.savefig(join(data_dir, analysis_dir, 'worker_gold_standard_task_accuracy.png'))
    plt.show()

# Bar graph of average worker performance on all categories
def worker_accuracy_bar_graph(df):
    xs = ['WMC', 'WMI', 'NWM', 'Total']
    keys = [
        'WearingMaskCorrectlyAccuracy', 'WearingMaskIncorrectlyAccuracy', 
        'NotWearingMaskAccuracy', 'TotalAccuracy'
    ]
    # Compute weighted average of accuracies
    total_tasks = sum([i for i in df['TasksCompleted']])
    total = df.apply(
        lambda x: pd.Series([x['TasksCompleted']] + 
            [x['TasksCompleted'] * x[key] for key in keys]),
        axis = 1
    )
    total = total.apply(sum)
    # Plot bar graph
    ys = [total[i+1] / total[0] for i,_ in enumerate(keys)]
    plt.bar(xs, ys, color=['green', 'yellow', 'red', 'black'])
    plt.title('Average worker accuracy across all mask categories')
    plt.xlabel('Mask Categories')
    plt.ylabel('Percentage Accuracy')
    plt.ylim((0, 1))
    plt.savefig(join(data_dir, analysis_dir, 'worker_gold_standard_accuracies.png'))
    plt.show()

# Scatter Plot of Time spent on each task vs Accuracy
# Color map that highlights minimum accuracy of each worker across all categories
def worker_time_accuracy_scatter_plot(df):
    xs = []
    ys = []
    cs = []
    max_time = 0
    for _, row in df.iterrows():
        # Append time and accuracy
        time = row['TimePerTask']
        if time > max_time:
            max_time = time
        xs.append(time)
        ys.append(row['TotalAccuracy'])
        # Find least accurate category
        min_acc = min([
            row['WearingMaskCorrectlyAccuracy'], 
            row['WearingMaskIncorrectlyAccuracy'],
            row['NotWearingMaskAccuracy']
        ])
        cs.append(min_acc)
    # Plot graph
    cmap = LinearSegmentedColormap.from_list("", ["red","yellow","green"])
    plt.scatter(xs, ys, c=cs, cmap=cmap, vmin=0, vmax=1)
    plt.title("Average time spent on each task vs Accuracy of Workers")
    plt.xlabel("Average time spent on each task (s)")
    plt.ylabel("Percentage Accuracy")
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Minimum accuracy across all mask categories', rotation=270)
    plt.savefig(join(data_dir, analysis_dir, 'worker_gold_standard_time_accuracy.png'))
    plt.show()

def main():
    # Read true labels into dictionary
    df = pd.read_csv(join(data_dir, analysis_dir, 'gold_standard_quality.csv'))
    worker_task_accuracy_scatter_plot(df)
    worker_accuracy_bar_graph(df)
    worker_time_accuracy_scatter_plot(df)

if __name__ == '__main__':
    main()