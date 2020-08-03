 # Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compute scores for classification"""

import argparse
import csv
import os

import numpy as np


def main(file_path):
    targets = []
    predictions = []
    with open(file_path) as f: 
        read_tsv = csv.reader(f, delimiter="\t")
        for line in read_tsv:
            targets.append(int(line[-1]))
            predictions.append(int(line[-2]))
    
    number_of_samples = len(predictions)
    targets = np.array(targets)
    predictions = np.array(predictions)
    overall_accuracy = np.sum(np.equal(targets, predictions))/number_of_samples
    print("Accuracy: {:.4f}".format(overall_accuracy)) 
    
    for category in np.unique(targets):
        precision, recall, f1 = _calculate_recall_precision(predictions, targets, 
                                                            category)
        print("----- For category", category, "-----")
        print("Precision is {:.4f}".format(precision))
        print("Recall is {:.4f}".format(recall))
        print("F1 is {:.4f}".format(f1))
    
    
def _calculate_recall_precision(predictions, targets, positive_class):
    '''Calculate the recall and precision score of binary classification.
    Args:
      predictions: a list of predictions
      targets: a list of targets
      positive_class: the class that is considered as the positive class
    
    Returns:
      precision: true positive / (true positive + false positive)
      recall: true positive / (true positive + false negative)
      F1 score: 2 * precision * recall / (precision + recall)
    '''
    true_positive = 0
    false_positive = 0
    false_negative = 0
    
    for index, prediction in enumerate(predictions):
        if targets[index] == positive_class:
            if prediction == positive_class:
                true_positive += 1
            else:
                false_negative += 1
        elif prediction == positive_class:
            false_positive += 1
            
    if true_positive + false_positive != 0:
        precision = true_positive/(true_positive + false_positive)
    else: 
        precision = float('nan')
        print("All samples are predicted not to be", positive_class)
        
    if true_positive + false_negative != 0:
        recall = true_positive/(true_positive + false_negative)
    else:
        recall = float("nan")
        print("None of the samples are predicted to be class", 
              positive_class)
    score_f1 = 2*precision*recall/(precision + recall)
    
    return precision, recall, score_f1
    

if __name__ == "__main__":
    """
    Calculate overall accuracy and accuracy by category for the classifier.
    
    usage: score_main.py [-h] pred_file

    positional arguments:
     pred_file   Absolute path to the prediction file.

    optional arguments:
     -h, --help  show this help message and exit
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_file", help="Absolute path to the prediction file.")    
    arguments = parser.parse_args()
    
    pred_file_path = os.path.expanduser(arguments.pred_file)
    if not os.path.isfile(pred_file_path):
        raise ValueError("Cannot find the prediction file.")
        
    main(pred_file_path)

