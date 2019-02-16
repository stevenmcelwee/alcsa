from alcsa import ActiveLearner
import numpy as np
from dataset import UNSW
import pandas as pd
from scipy.stats import binom

input_data_path = 'datasets/UNSW-NB15/UNSW-NB15_1.csv'
output_data_path = 'results/experiment_02.csv'
min_attack_labels = 1
num_runs = 10

# Experiment 2 - Evaluate the number of observations needed to identify computer systems under attack

# Create a reference copy of the original dataframe for analysis with observations
dataset = UNSW(input_data_path)
original_df = dataset.get_df()
output_cols = ['srcip']
original_df = original_df[output_cols].copy()
count_df = original_df['srcip'].value_counts().reset_index()
count_df.columns = ['srcip','count']

cols_to_sum = []
# Create a series of num_runs observations using active learning with bagging
for i in range(num_runs):
    print("Run %s..." % (i))
    al = ActiveLearner(input_data_path)
    count_attacks = 0
    Y = None
    retries = -1
    while count_attacks < min_attack_labels:
        sample_df = al.create_sample(query_method='bagging')
        Y = al.query_oracle(sample_df)
        count_attacks = sum(Y)
        retries += 1
    al.classify(sample_df, Y)
    # Append observations to original_df
    feature_name = 'P%s' % (i)
    observations_df = al.get_df()
    original_df[feature_name] = observations_df['P']
    # Calculate the cumulative sum of observations
    cols_to_sum.append(feature_name)
    sum_name = 'S%s' % (i)
    output_cols.append(sum_name)
    original_df[sum_name] = original_df[cols_to_sum].sum(axis=1)

results_df = original_df[output_cols].groupby(['srcip']).sum().reset_index()
final_df = pd.merge(left=count_df, right=results_df, left_on ='srcip', right_on='srcip')
print(final_df)
final_df.to_csv(output_data_path, index=False)
# Calculate probability mass function separately in Excel






