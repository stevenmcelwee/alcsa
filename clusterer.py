from sklearn.cluster import KMeans
import random
import numpy as np
from collections import Counter

class PartitionGenerator:

    def __init__(self, dataset, num_partitions, min_feature_ratio=0.25, max_feature_ratio=0.75, min_clusters=40, max_clusters=100):
        self.__labels = []  # store labels for each partition in a list

        feature_columns = dataset.get_feature_cols()
        max_features = int(round(max_feature_ratio*len(feature_columns),0))
        min_features = int(round(min_feature_ratio*len(feature_columns),0))
        print("Generating bagging plans...")
        bagging_plans = self.__gen_bagging_plan(feature_columns, num_partitions, max_features, min_features)
        i = 0
        for plan in bagging_plans:
            # generate settings for kmeans
            n_clusters = np.random.randint(min_clusters, max_clusters)
            print "Clustering %s: %s features into %s clusters..." % (i, len(plan), n_clusters)
            print plan
            # create the model
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
            df = dataset.get_df()
            X = df[plan]
            # fit the model
            model.fit(X)
            # append the labels
            # TODO: Can I remove self.__bagging_plans from the label object? I added a return for it.
            self.__labels.append([plan, model.labels_])
            i += 1

    def __gen_bagging_plan(self, cols, p_count, max_features, min_features):
        # create a sampling plan for bagging
        bagging_plan = []
        for p in range(p_count):
            num_features_this_sample = random.randint(min_features, max_features)
            features_this_sample = []
            for i in range(num_features_this_sample):
                features_this_sample.append(np.random.choice(cols))
            bagging_plan.append(features_this_sample)
        return bagging_plan

    def get_labels(self):
        return self.__labels

    def get_bagging_plan(self):
        return self.__bagging_plans

    # Do I need this function? probably can be deleted
    def eval_partitions(self):
        i = 0
        for cluster_labels in self.__labels:
            num_labels = len(cluster_labels)*1.0    # get total number of labels for calculating the ratio
            col_label = "P%s" % (i)
            counter = Counter()
            for label in cluster_labels:
                counter[label] += 1
            i += 1
            # enumerate the unique elements in the counter
            eval_object = {}
            for label in list(counter):
                percentage = round(counter[label]/num_labels, 3)
                # print "Total labels: %s; this label: %s; result: %s" % ( num_labels, counter[label], percentage)
                eval_object[label] = percentage

            values = []
            for key in eval_object.keys():
                values.append(eval_object[key])
            arr = np.array(values)
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            min = mean - 2*std
            max = mean + 2*std

            print col_label, eval_object, min, max

class Evaluator:

    def __init__(self, partition_generator):
        self.__pg = partition_generator

        self.__eval_object = []

        # Get labels for list: 1) partition; 2) bagging plan_list, labels_list
        self.__labels = partition_generator.get_labels()

        # Loop through the partitions
        for partition in self.__labels:
            # count the number of each label in the partition
            counter = Counter()
            for label in partition[1]:
                counter[label] += 1
            # loop through counter object to get some statistics
            # first convert counter to a list
            label_counts = []
            for key, value in counter.iteritems():
                label_counts.append(value)
            self.__eval_object.append({ "counter": counter,
                                        "total": sum(label_counts),
                                        "label_count": len(label_counts),
                                        "mean": np.mean(label_counts, axis=0),
                                        "stdev": np.std(label_counts, axis=0),
                                        "features": partition[0]})

    def get_anoms(self, distance):
        # holder for 2D list of partitions and labels
        parition_anom_labels = []
        # loop through all of the partitions to calculate boundaries and generate list of anomalies
        for partition in self.__labels:
            # calculate boundaries
            counter = Counter()
            for label in partition:
                counter[label] +=1
            label_counts = []
            for key, value in counter.iteritems():
                label_counts.append(value)
            min_value = np.mean(label_counts, axis=0) - distance * np.std(label_counts, axis=0)
            max_value = np.mean(label_counts, axis=0) + distance * np.std(label_counts, axis=0)

            # evaluate each label in counter to see if it is an anomaly
            # store lookup in dictionary
            anoms = {}
            # approach 1 - above and below x stdev from mean
            # for key, value in counter.iteritems():
            #     if value > max_value:
            #         anoms[key] = 1
            #     elif value < min_value:
            #         anoms[key] = 1
            #     else:
            #         anoms[key] = 0
            # approach 2 - only below x stdev from mean
            for key, value in counter.iteritems():
                if value < min_value:
                    anoms[key] = 1
                else:
                    anoms[key] = 0

            # Now create new list of labels as anomaly = 1 or normal = 0
            anom_labels = []
            for label in partition:
                anom_labels.append(anoms[label])

            # append the results of this partition to the master place holder
            parition_anom_labels.append(anom_labels)

        return parition_anom_labels



    def print_eval_object(self):
        partition_number = 0
        for partition in self.__eval_object:
            anom1 = 0
            anom2 = 0
            anom3 = 0
            anom1_label = []
            anom2_label = []
            anom3_label = []
            anom1_max = partition["mean"] + partition["stdev"]
            anom1_min = partition["mean"] - partition["stdev"]
            anom2_max = partition["mean"] + 2*partition["stdev"]
            anom2_min = partition["mean"] - 2*partition["stdev"]
            anom3_max = partition["mean"] + 3*partition["stdev"]
            anom3_min = partition["mean"] - 3*partition["stdev"]
            for key, value in partition["counter"].iteritems():
                if value > anom3_max:
                    anom3 +=1
                    anom3_label.append(key)
                if value < anom3_min:
                    anom3 +=1
                    anom3_label.append(key)
                if value > anom2_max:
                    anom2 +=1
                    anom2_label.append(key)
                if value < anom2_min:
                    anom2 +=1
                    anom2_label.append(key)
                if value > anom1_max:
                    anom1 +=1
                    anom1_label.append(key)
                if value < anom1_min:
                    anom1 +=1
                    anom1_label.append(key)

            print partition_number, partition["label_count"], anom1, anom1_label, anom1_min, anom1_max, \
                anom2, anom2_label, anom2_min, anom2_max, \
                anom3, anom3_label, anom3_min, anom3_max, partition["counter"], partition["features"]



