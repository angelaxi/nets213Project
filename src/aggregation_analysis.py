from result_process import em_vote, worker_quality
import pandas as pd

data_dir = '../data/'
analysis_dir = 'analysis'

def main():
    # Read in CSV result file with pandas
    result_df = pd.read_csv(join(data_dir, 'classification_hit_output.csv'))
    # Compute worker quality and confusion matrix from gold standard labels
    _, cm = worker_quality(result_df)
    # 1 iteration EM with gold standard label performance as initial quality
    unconverged_weighted_labels = em_vote(result_df, cm, 1, return_dict=True)
    # Converged EM with gold standard label performance as initial quality
    converged_weighted_labels = em_vote(result_df, cm, -1, return_dict=True)
    # 1 iteration EM assuming all workers are initially perfect
    unconverged_unweighted_labels = em_vote(result_df, None, 1, return_dict=True)
    # Converged EM assuming all workers are initially perfect
    converged_unweighted_labels = em_vote(result_df, None, -1, return_dict=True)

