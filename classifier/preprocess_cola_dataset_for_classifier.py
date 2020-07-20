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

import preprocess_utils
""" Preprocess the CoLA (The Corpus of Linguistic Acceptability) grammar 
dataset classification task."""

PREPROCESSED_FILE_PATH = "~/classifier_preprocessed_cola_dataset.tsv"
MIXED_FILE_PATH = "~/classifier_mixed_training_set_grammar.tsv"


def main(args):
    """Preprocess the CoLA grammar dataset.
    Args:
        args: command line arguments.
    """
    data_file = os.path.expanduser(args.raw_data_file)
    if not os.path.isfile(data_file):
        raise Exception("Data file not found.")
    
    sentences_positive = []
    sentences_negative = []
    
    with open(data_file) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for line in read_tsv:
            if int(line[1]) == 1:
                sentences_positive.append(line[3])
            else:
                sentences_negative.append(line[3])

    cleaned_sentences_positive = preprocess_utils.text_strip(sentences_positive)
    cleaned_sentences_negative = preprocess_utils.text_strip(sentences_negative)

    print("Number of samples is", 
          len(cleaned_sentences_positive) + len(cleaned_sentences_negative))
    print("Number of incorrect sample is", len(cleaned_sentences_negative),
         "and number of correct sample is", len(cleaned_sentences_positive))

    spaced_sentences_positive = preprocess_utils.tokenize_with_space(
        cleaned_sentences_positive)
    spaced_sentences_negative = preprocess_utils.tokenize_with_space(
        cleaned_sentences_negative)
    
    with open(os.path.expanduser(PREPROCESSED_FILE_PATH), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for positive_sentence in spaced_sentences_positive:
            tsv_writer.writerow([positive_sentence, "1"])
        for negative_sentence in spaced_sentences_negative:
            tsv_writer.writerow([negative_sentence, "0"])
    print("-------Preprocessed data saved to", PREPROCESSED_FILE_PATH, "-------")

    print("-------Now mixing dataset with the MS dataset.-------")
    MS_data_file = os.path.expanduser(args.MS_data_file)
    if not os.path.isfile(MS_data_file):
        raise Exception("Microsoft data file not found.")
        
    MS_sentences = []
    MS_ratings = []
    number_of_MS_samples_in_each_category = [0, 0]

    with open(MS_data_file) as tsv_file:
        read_tsv = csv.reader(tsv_file, delimiter="\t")  
        for line in read_tsv:
            MS_sentences.append(line[0])
            MS_ratings.append(int(line[1]))
            number_of_MS_samples_in_each_category[int(line[1])] += 1
    
    max_negative_rate = (number_of_MS_samples_in_each_category[0] + 
                         len(cleaned_sentences_negative))/ \
    (sum(number_of_MS_samples_in_each_category) + len(cleaned_sentences_negative))
    min_negative_rate = (number_of_MS_samples_in_each_category[0] + 
                         len(cleaned_sentences_negative))/  \
    (sum(number_of_MS_samples_in_each_category) + len(cleaned_sentences_positive) +
    len(cleaned_sentences_negative))
    
    goal_percentage = args.goal_percentage_of_neg_samples
    if goal_percentage is None:
        number_of_pos_sample_to_include = 0
    else:
        if goal_percentage > max_negative_rate:
            raise Exception("The goal negative sample percentage is greater than the largest"
                           "possible value {:.2f}".format(max_negative_rate))
            
        if goal_percentage < min_negative_rate:
            raise Exception("The goal negative sample percentage is smaller than the smallest"
                           "possible value {:.2f}".format(min_negative_rate))
        
        number_of_pos_sample_to_include = int((1 - goal_percentage)/goal_percentage * 
                                              (len(cleaned_sentences_negative) + 
                                               number_of_MS_samples_in_each_category[0]) - 
                                             number_of_MS_samples_in_each_category[1])

        print("------- Including", number_of_pos_sample_to_include, "samples from the cola dataset.")
        
    MS_sentences = MS_sentences + spaced_sentences_positive[0:number_of_pos_sample_to_include] + \
                   spaced_sentences_negative
    MS_ratings = MS_ratings + [1] * number_of_pos_sample_to_include + [0] * len(spaced_sentences_negative)

    actual_negative_rate = (number_of_MS_samples_in_each_category[0] + 
                            len(spaced_sentences_negative)) / \
                            (sum(number_of_MS_samples_in_each_category) + 
                            len(spaced_sentences_negative) + number_of_pos_sample_to_include)
                                                                  
    print("-------The percentage of negative sample is", "{:.2f}".format(actual_negative_rate), 
          "-------")
    
    shuffled_index = list(range(len(MS_sentences)))
    random.shuffle(shuffled_index)
    
    with open(os.path.expanduser(MIXED_FILE_PATH), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for index in shuffled_index:
            tsv_writer.writerow([MS_sentences[index], MS_ratings[index]])
            
    print("-------", len(MS_sentences), "samples saved to", MIXED_FILE_PATH, "-------")

    
if __name__ == "__main__":
    """
    Preprocess the CoLa grammar rating dataset and mix with Microsoft dataset.
    
    CoLa data needs to be downloaded from https://nyu-mll.github.io/CoLA and the abosolute path to the raw data file is 
    provided as a command line argument.
    The CoLa dataset is mixed with the Microsoft datase whose data file path is also provided as a command line argument.
    All negative samples in the CoLa dataset are mixed into the final training set, while the number of CoLa positive 
    samples mixed is adjusted to match the target percentage of negative samples if the target is provided. If the target
    percentage is not provided, none of the positive samples will be mixed.
    Dataset is split into training, tuning, and validation sets, with the number of samples in the tuning and validation
    set being specified in the command line argument. The three sets are saved in three separate tsv files, and all the
    preprocessed data are saved in another tsv file.
    
    usage: preprocess_cola_dataset_for_classifier.py [-h] [-goal_percentage_of_neg_samples GOAL_PERCENTAGE_OF_NEG_SAMPLES]
           raw_data_file MS_data_file

    positional arguments:
      raw_data_file         Absolute path to the cola grammar dataset tsv file.
      MS_data_file          Absolute path to the preprocessed MS dataset tsv file.

    optional arguments:
      -h, --help            show this help message and exit
      -goal_percentage_of_neg_samples GOAL_PERCENTAGE_OF_NEG_SAMPLES
                            The goal of negative samplepercentage after mixing the MS and cola dataset. If not provided, 
                            the mixing will maximize the percentage of negative samples.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_file", help="Absolute path to the cola grammar dataset tsv file.")
    parser.add_argument("MS_data_file", help="Absolute path to the preprocessed MS dataset tsv file.")
    parser.add_argument('-goal_percentage_of_neg_samples', type=float, help="The goal of negative sample"
                       "percentage after mixing the MS and cola dataset. If not provided, the mixing "
                       "will maximize the percentage of negative samples.")
    arguments = parser.parse_args()
    main(arguments)
