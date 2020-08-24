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
    
    
def calculate_recall_precision(predictions, targets, positive_class):
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
    
    if true_positive == 0:
        precision = 0
    elif true_positive + false_positive != 0:
        precision = true_positive/(true_positive + false_positive)
    else: 
        precision = float('nan')
        print("None of the samples are predicted to be class", positive_class)
        
    
    if true_positive == 0:
        recall = 0
    elif true_positive + false_negative != 0:
        recall = true_positive/(true_positive + false_negative)
    else:
        recall = float("nan")
        print("None of the samples are predicted to be class", positive_class)
    
    if precision + recall == 0:
        score_f1 = 0
    elif precision == float("nan") or recall == float("nan"):
        score_f1 = float("nan")
    else:
        score_f1 = 2*precision*recall/(precision + recall)
    
    return precision, recall, score_f1
