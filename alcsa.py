import pandas as pd
from dataset import UNSW
from sklearn.cluster import KMeans
from random import sample
from random import randint
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class ActiveLearner:
    def __init__(self, input_file_path):
        # Load the dataset as a DataFrame
        self.dataset = UNSW(input_file_path)
        self.dataset.preprocess()
        self.df = self.dataset.get_df()

    def get_df(self):
        return self.df

    def create_sample(self, query_method):
        # Create the sample. The output of this if block is a sample_df
        sample_df = None
        if query_method == 'random':
            sample_df = self.get_random_sample(self.df)
        elif query_method == 'kmeans':
            sample_df = self.get_kmeans_sample(self.df)
        elif query_method == 'bagging':
            sample_df = self.get_bagging_sample(self.df)
        return sample_df

    def query_oracle(self, sample_df):
        # Ask the oracle for values - it's already in the sample_df
        Y = sample_df['Label'].values
        return Y

    def classify(self, sample_df, Y):
        # Train a Random Forest classifier using the sample_df as the trining dataset
        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        X = sample_df[self.dataset.get_feature_cols()].values
        clf.fit(X, Y)
        # Label unknown labels in dataset that match oracle values
        X = self.df[self.dataset.get_feature_cols()].values
        pred_label = "P"
        self.df[pred_label] = clf.predict(X)

    def get_random_sample(self, df):
        sample_df = df.sample(40)
        return sample_df

    def get_kmeans_sample(self, df):
        n_clusters = 40
        model = KMeans( n_clusters = n_clusters,
                        init="k-means++",
                        n_init=10,
                        max_iter=300,
                        tol=1e-4,
                        precompute_distances="auto",
                        verbose=0,
                        random_state=None,
                        copy_x=True )
        X = df[self.dataset.get_feature_cols()].values
        # fit the model
        model.fit(X)
        # append labels to original dataframe as new column
        df['cid'] = model.labels_
        # Create a sample from among the cluster ids
        sample_df = None
        for cid in df['cid'].unique():
            if sample_df is None:
                sample_df = df.loc[df['cid']==cid].sample(1)
            else:
                sample_df = sample_df.append(df.loc[df['cid']==cid].sample(1))
        sample_df = sample_df.drop(columns=['cid'])
        return sample_df

    def get_bagging_sample(self, df):
        # Define attributes for bagging with mins and maxes
        min_features = 20
        max_features = 35
        min_clusters = 30
        max_clusters = 50
        num_random_columns = randint(min_features, max_features)
        columns = sample(self.dataset.get_feature_cols(),num_random_columns)
        cluster_df = df[columns]
        n_clusters = randint(min_clusters, max_clusters)
        # print("Pass %s - columns: %s; clusters: %s" % (i, num_random_columns, n_clusters))
        model = KMeans( n_clusters = n_clusters,
                        init="k-means++",
                        n_init=10,
                        max_iter=300,
                        tol=1e-4,
                        precompute_distances="auto",
                        verbose=0,
                        random_state=None,
                        copy_x=True )
        # generate feature dataset
        df = self.dataset.get_df()
        X = df[columns].values
        # fit the model
        model.fit(X)
        # append labels to original dataframe as new column
        df['cid'] = model.labels_

        # for each cluster value, loop to create a sample to send to the oracle
        sample_df = None
        for cid in df['cid'].unique():
            if sample_df is None:
                sample_df = df.loc[df['cid']==cid].sample(1)
            else:
                sample_df = sample_df.append(df.loc[df['cid']==cid].sample(1))
        sample_df = sample_df.drop(columns=['cid'])
        return sample_df

    def eval(self, description):
        df = self.df
        # Evaluate results
        conditions = [(df['Label'] == 0) & (df['P'] == 0),
                      (df['Label'] == 0) & (df['P'] == 1),
                      (df['Label'] == 1) & (df['P'] == 0),
                      (df['Label'] == 1) & (df['P'] == 1)]
        choices = ['TN', 'FP', 'FN', 'TP']
        df['R'] = np.select(conditions, choices)

        result_df = df[['P','R']].groupby('R').count()
        result_df = result_df.T
        if not 'FN' in result_df.columns:
            result_df['FN'] = 0
        if not 'FP' in result_df.columns:
            result_df['FP'] = 0
        if not 'TN' in result_df.columns:
            result_df['TN'] = 0
        if not 'TP' in result_df.columns:
            result_df['TP'] = 0

        result_df['Accuracy'] = 1.0*(result_df['TP']+result_df['TN'])/(result_df['TP']+result_df['TN']+result_df['FP']+result_df['FN'])
        result_df['TPR'] = 1.0*(result_df['TP'])/(result_df['TP']+result_df['FN'])
        result_df['FPR'] = 1.0*(result_df['FP'])/(result_df['FP']+result_df['TN'])

        result_df['Description'] = description

        return result_df
