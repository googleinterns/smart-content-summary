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
import argparse
import csv
import os
import random

import numpy as np


""" Mix synthetic negative samples into training data."""


def combine_two_tsv_files(original_file_path, append_file_path, number_of_new_lines,
                         output_file_path):
    """Mix some of the lines in a file to another file, and write to a target path.
    Args:
      original_file_path: the file where all the lines will be present in the 
        target file
      append_file_path: the file where some of the lines will be present in the target 
        file
      number_of_new_lines: the number of lines from the second file that will be 
        present in the target file
      output_file_path: the path where the target file will be saved
    """
    with open(original_file_path) as f:
        original_lines = f.readlines()
    with open(append_file_path) as f:
        new_lines = f.readlines()
    
    if number_of_new_lines < 0:
        raise ValueError("The target ratio is too small.")
    if number_of_new_lines > len(new_lines):
        raise ValueError("The target ratio is too big given the number of synthetic data.")
    
    # Grammar classifier only needs one input while meaning classifier needs two
    number_of_inputs = len(original_lines[0].rstrip().split("\t")) - 1
    
    out_file = open(os.path.expanduser(output_file_path), 'wt')
    tsv_writer = csv.writer(out_file, delimiter='\t') 
    
    # Reshuffle the dataset after mixing in new sample.
    random.shuffle(new_lines)
    shuffled_index = list(range(len(original_lines) + number_of_new_lines))
    random.shuffle(shuffled_index)
    for index in shuffled_index:
        if index >= len(original_lines):
            new_line = new_lines.pop().rstrip().split("\t")[- number_of_inputs:]
            new_line.append("0")
        else:
            new_line = original_lines[index].rstrip().split("\t")           
        tsv_writer.writerow(new_line)
    out_file.close()
    print("----- Data file saved to", output_file_path, "-----")
            

def calculate_number_of_synthetic_data_to_mix(original_data_file, target_ratio):
    """Calculate the number of negative samples that need to be added to 
    achieve the target ratio of negative samples.
    
    Args:
      original_data_file: path to the original data file
      target_ratio: the target ratio of negative samples
    
    Returns:
      The number of negative samples needed to achieve the target ratio.
    """
    total_number_of_samples = 0
    original_negative_sample_count = 0
    tsv_file = open(original_data_file)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for line in read_tsv:
        if int(line[-1]) == 0:
            original_negative_sample_count += 1
        total_number_of_samples += 1
    tsv_file.close()
    return int((original_negative_sample_count - 
            total_number_of_samples * target_ratio)/(target_ratio - 1))
        
    
def main(args):
    """Mix synthetic negative samples into training data.
    Args:
        args: command line arguments.
    """
    synthetic_data_file = os.path.expanduser(args.synthetic_data_file)
    if not os.path.isfile(synthetic_data_file):
        raise Exception("Synthetic data file not found.")
    
    original_data_file = os.path.expanduser(args.original_data)
    if not os.path.isfile(original_data_file):
        raise Exception("Original data file not found.")
    
    target_ratio = float(args.target_ratio)
    
    combine_two_tsv_files(original_data_file, synthetic_data_file, 
                          calculate_number_of_synthetic_data_to_mix(
                              original_data_file, target_ratio),
                         args.output_file_path)
    
    
if __name__ == "__main__":
    """
    Mix synthetic data with original dataset to achieve desirable ratio of negative samples. 
    
    usage: mixing_in_negative_samples.py [-h]
                                     synthetic_data_file original_data
                                     target_ratio output_file_path

    positional arguments:
      synthetic_data_file  Absolute path to synthetic dataset.
      original_data        Absolute path to the original data.
      target_ratio         The target ratio of negative sample.
      output_file_path     Path to the output file

    optional arguments:
      -h, --help           show this help message and exit
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("synthetic_data_file", help="Absolute path to synthetic dataset.")
    parser.add_argument("original_data", help="Absolute path to the original data.")
    parser.add_argument('target_ratio', type=float, help="The target ratio of negative sample.")
    parser.add_argument('output_file_path', help="Path to the output file")
    arguments = parser.parse_args()
    main(arguments)
