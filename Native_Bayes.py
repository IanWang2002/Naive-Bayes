import math
from typing import List

num_C = 7  # Total number of classes

class Solution:

    def prior(self, X_train: List[List[int]], Y_train: List[int]) -> List[float]:
        """Calculate the prior probabilities of each class
        Args:
          X_train: Row i represents the i-th training datapoint
          Y_train: The i-th integer represents the class label for the i-th training datapoint
        Returns:
          A list of length num_C where num_C is the number of classes in the dataset
        """
    # implement this function
        class_counts = [0] * num_C
        N = len(Y_train)

        for y in Y_train:
            class_counts[y - 1] += 1

        priors = []
        for count in class_counts:
            prob = (count + 0.1) / (N + 0.1 * num_C)
            priors.append(prob)

        return priors

    def label(self, X_train: List[List[int]], Y_train: List[int], X_test: List[List[int]]) -> List[int]:
        """Calculate the classification labels for each test datapoint
        Args:
          X_train: Row i represents the i-th training datapoint
          Y_train: The i-th integer represents the class label for the i-th training datapoint
          X_test: Row i represents the i-th testing datapoint
        Returns:
          A list of length M where M is the number of datapoints in the test set
        """        
        num_features = len(X_train[0])
        N = len(X_train)

        # Get all unique values for each feature
        feature_values = [set() for _ in range(num_features)]
        for row in X_train:
            for i, val in enumerate(row):
                feature_values[i].add(val)
        num_feature_values = [len(s) for s in feature_values]

        # Initialize counts
        class_counts = [0] * num_C
        feature_counts = [ [dict() for _ in range(num_features)] for _ in range(num_C) ]

        for i in range(N):
            x = X_train[i]
            y = Y_train[i] - 1  # Convert to 0-based class index
            class_counts[y] += 1

            for j in range(num_features):
                value = x[j]
                if value not in feature_counts[y][j]:
                    feature_counts[y][j][value] = 1
                else:
                    feature_counts[y][j][value] += 1

        # Compute log priors
        log_prior = []
        for c in range(num_C):
            logp = math.log((class_counts[c] + 0.1) / (N + 0.1 * num_C))
            log_prior.append(logp)

        # Predict
        predictions = []
        for x in X_test:
            log_probs = []
            for c in range(num_C):
                logp = log_prior[c]
                total_class = class_counts[c]
                for j in range(num_features):
                    value = x[j]
                    count = feature_counts[c][j].get(value, 0)
                    prob = (count + 0.1) / (total_class + 0.1 * num_feature_values[j])
                    logp += math.log(prob)
                log_probs.append(logp)
            predictions.append(log_probs.index(max(log_probs)) + 1)

        return predictions