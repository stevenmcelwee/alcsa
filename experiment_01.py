from alcsa import ActiveLearner
import numpy as np

input_data_path = 'datasets/UNSW-NB15/UNSW-NB15_1.csv'
output_data_path = 'results/experiment1.csv'
min_attack_labels = 1
num_runs_per_method = 30

# Experiment 1 - Compare accuracy with different query methods: 1) random; 2) clustering; and 3) clustering with bagging
# Hypothesis 1 - Clustering with bagging improves query effectiveness for active learning.
results_df = None

# Do a batch of random selection
for i in range(num_runs_per_method):
    run_name = 'Random'
    al = ActiveLearner(input_data_path)
    count_attacks = 0
    Y = None
    retries = -1
    while count_attacks < min_attack_labels:
        sample_df = al.create_sample(query_method='random')
        Y = al.query_oracle(sample_df)
        count_attacks = sum(Y)
        retries += 1
    print("Run %s: %s; retries = %s" % (i, run_name, retries))
    al.classify(sample_df, Y)
    if results_df is None:
        results_df = al.eval(run_name)
    else:
        results_df = results_df.append(al.eval(run_name))

# Do a batch of kmeans
for i in range(num_runs_per_method):
    run_name = 'k-Means'
    al = ActiveLearner(input_data_path)
    count_attacks = 0
    Y = None
    retries = -1
    while count_attacks < min_attack_labels:
        sample_df = al.create_sample(query_method='kmeans')
        Y = al.query_oracle(sample_df)
        count_attacks = sum(Y)
        retries += 1
    print("Run %s: %s; retries = %s" % (i, run_name, retries))
    al.classify(sample_df, Y)
    if results_df is None:
        results_df = al.eval(run_name)
    else:
        results_df = results_df.append(al.eval(run_name))

# Do a batch of bagging with kmeans
for i in range(num_runs_per_method):
    run_name = 'Bagging'
    al = ActiveLearner(input_data_path)
    count_attacks = 0
    Y = None
    retries = -1
    while count_attacks < min_attack_labels:
        sample_df = al.create_sample(query_method='bagging')
        Y = al.query_oracle(sample_df)
        count_attacks = sum(Y)
        retries += 1
    print("Run %s: %s; retries = %s" % (i, run_name, retries))
    al.classify(sample_df, Y)
    if results_df is None:
        results_df = al.eval(run_name)
    else:
        results_df = results_df.append(al.eval(run_name))

print(results_df)
results_df.to_csv(output_data_path, index=False)






