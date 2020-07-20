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
    print("Accuracy: {:.2f}".format(overall_accuracy))
    
    correct_counts_breakdown = {}
    total_counts_breakdown = {}
    
    for category in np.unique(targets):
        this_category_index = np.equal(targets, category)
        category_accuracy = np.sum(np.equal(targets[this_category_index], 
                                           predictions[this_category_index]))/ np.sum(this_category_index)
        print("Accuracy for category", category, "with", np.sum(this_category_index), 
              "samples is {:.2f}".format(category_accuracy))
    
    
        

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

