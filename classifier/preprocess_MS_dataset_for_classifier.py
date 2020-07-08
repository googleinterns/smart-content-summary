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
""" Preprocess the Microsoft text summarization dataset for classification task."""

GRADING_COMMENTS = ["Most important meaning Flawless language", "Most important meaning Minor errors", \
                    "Most important meaning Disfluent or incomprehensible", "Much meaning Flawless language", \
                    "Much meaning Minor errors", "Much meaning Disfluent or incomprehensible", \
                    "Little or none meaning Flawless language", "Little or none meaning Minor errors", \
                    "Little or none meaning Disfluent or incomprehensible"]
GRADING_NUMBER = ["6", "7", "9", "11", "12", "14", "21", "22", "24"]
GRAMMAR_GRADING_BUCKET = [["9", "14", "24"], ["7", "12", "22"], ["6", "11", "21"]]
MEANING_GRADING_BUCKET = [["21", "22", "24"], ["11", "12", "14"], ["6", "7", "9"]]

PREPROCESSED_FILE_PATH = "~/classifier_preprocessed_MS_dataset.tsv"
TRAIN_FILE_PATH_GRAMMAR = "~/classifier_train_MS_dataset_grammar.tsv"
TUNE_FILE_PATH_GRAMMAR = "~/classifier_tune_MS_dataset_grammar.tsv"
VALID_FILE_PATH_GRAMMAR = "~/classifier_valid_MS_dataset_grammar.tsv"
TRAIN_FILE_PATH_MEANING = "~/classifier_train_MS_dataset_meaning.tsv"
TUNE_FILE_PATH_MEANING = "~/classifier_tune_MS_dataset_meaning.tsv"
VALID_FILE_PATH_MEANING = "~/classifier_valid_MS_dataset_meaning.tsv"


def __process_row(row):
    """Split a row into the original sentence, its corresponding summary and its rating.

    Args:
        row: a row in the MS dataset tsv file.
    Returns:
        current_original_sentence: the original sentence of the row
        current_shortened_sentences_list: a list of summaries corresponding to the current_original_sentence
        grammar_ratings_list: a list of grammar ratings of the summaries in 
          current_shortened_sentences_list (0 being the worst and 2 being the best)
        meaning_ratings_list: a list of meaning ratings of the summaries in 
          current_shortened_sentences_list (0 being the worst and 2 being the best)
    """
    row_flattened = []
    for i in range(len(row)):
        splitted_row = row[i].split(" ||| ")
        for j in range(len(splitted_row)):
            if splitted_row[j] not in GRADING_COMMENTS:
                row_flattened.append(splitted_row[j])

    current_source = row_flattened[2]
    current_summary_list = []
    current_ratings_list = []

    this_summary = row_flattened[3]
    this_ratings = []
    for i in range(4, len(row_flattened)):
        if i + 1 == len(row_flattened):
            this_ratings.append(row_flattened[i])
            this_ratings = this_ratings[2:]
            if len(this_ratings) != 0:
                current_summary_list.append(this_summary)
                current_ratings_list.append(this_ratings)

        elif not row_flattened[i].isnumeric() and not row_flattened[i].split(";")[0].isnumeric():
            this_ratings = this_ratings[2:]
            if len(this_ratings) != 0:
                current_summary_list.append(this_summary)
                current_ratings_list.append(this_ratings)
                this_summary = row_flattened[i]
                this_ratings = []
        else:
            this_ratings.append(row_flattened[i])
    assert (len(current_summary_list) == len(current_ratings_list))
    
    grammar_ratings_list, meaning_ratings_list = __find_grammar_meaning_ratings(
        current_ratings_list)
    
    return current_source, current_summary_list, grammar_ratings_list, meaning_ratings_list


def __find_grammar_meaning_ratings(ratings_list):
    """ Given a list of ratings lists, find the grammar and meaning rating.
    
    Args:
      ratings_list: a list of list of ratings. 
    Returns:
      grammar_ratings: a list of grammar ratings
      meaning_ratings: a list of meaning ratings
    """
    grammar_ratings = []
    meaning_ratings = []
    for rating_list in ratings_list:
        grammar_ratings.append(
            __find_most_common_bucket(rating_list, GRAMMAR_GRADING_BUCKET))
        meaning_ratings.append(
            __find_most_common_bucket(rating_list, MEANING_GRADING_BUCKET))
    return grammar_ratings, meaning_ratings
        
    
def __find_most_common_bucket(items_list, buckets_list):
    """ Given a list of buckets and a list of items, find the index of the bucket where 
    most items are in.
    
    Args:
      items_list: a list of objects
      buckets_list: a list of object list ("bucket"). 
    Returns:
      The index of the bucket where most items are in.
    """
    bucket_item_count = np.zeros(len(buckets_list))
    
    for item in items_list:
        for bucket_index, bucket in enumerate(buckets_list):
            if item in bucket:
                bucket_item_count[bucket_index] += 1
    return np.argmax(bucket_item_count)
    
def __process_file(file_path):
    """Process a tsv file in the MS dataset.

    Args:
        file_path: direct path to the tsv file
    Returns:
        sentences: a list of original sentences
        summaries: a list of summaries corresponding to the original sentences
        grammar_ratings: a list of grammar ratings of the summaries
        meaning_ratings: a list of grammar ratings of the summaries
    """
    tsv_file = open(os.path.expanduser(file_path))
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    sentences = []
    summaries = []
    grammar_ratings = []
    meaning_ratings = []
    for row in read_tsv:
        row_sentence, row_summary, row_grammar, row_meaning = __process_row(row)

        for i in range(len(row_summary)):
            sentences.append(row_sentence)
            summaries.append(row_summary[i])
            grammar_ratings.append(row_grammar[i])
            meaning_ratings.append(row_meaning[i])

    return sentences, summaries, grammar_ratings, meaning_ratings


def main(args):
    """Preprocess the Microsoft text summarization dataset.

    Args:
        args: command line arguments.
    """
    data_dir = args.raw_data_dir
    if not os.path.isdir(os.path.expanduser(data_dir)):
        raise Exception("Data directory not found.")

    num_of_tuning_sam = args.num_of_tuning
    num_of_valid_sam = args.num_of_validation

    if num_of_valid_sam < 0 or num_of_tuning_sam < 0:
        raise Exception("Number of samples must be non-negative integers")


    train_data_file = data_dir + "/train.tsv"
    train_sentences, train_summaries, train_grammar, train_meaning = __process_file(train_data_file)
    test_data_file = data_dir + "/test.tsv"
    test_sentences, test_summaries, test_grammar, test_meaning = __process_file(test_data_file)
    valid_data_file = data_dir + "/valid.tsv"
    valid_sentences, valid_summaries, valid_grammar, valid_meaning = __process_file(valid_data_file)

    tot_sentences = train_sentences + test_sentences + valid_sentences
    tot_summaries = train_summaries + test_summaries + valid_summaries
    tot_grammar = train_grammar + test_grammar + valid_grammar
    tot_meaning = train_meaning + test_meaning + valid_meaning

    cleaned_sentences = preprocess_utils.text_strip(tot_sentences)
    cleaned_summaries = preprocess_utils.text_strip(tot_summaries)

    cleaned_sentences, cleaned_summaries = preprocess_utils.delete_empty_entry(cleaned_sentences, cleaned_summaries)
    preprocess_utils.validate_dataset(cleaned_sentences, cleaned_summaries)
    print("Number of samples is", len(cleaned_sentences))

    spaced_sentences = preprocess_utils.tokenize_with_space(cleaned_sentences)
    spaced_summaries = preprocess_utils.tokenize_with_space(cleaned_summaries)

    with open(os.path.expanduser(PREPROCESSED_FILE_PATH), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for i in range(len(spaced_sentences)):
            tsv_writer.writerow([spaced_sentences[i], spaced_summaries[i], tot_grammar[i], tot_meaning[i]])
    print("-------Preprocessed data saved to", PREPROCESSED_FILE_PATH, "-------")

    print("-------Now splitting dataset.-------")
    if num_of_tuning_sam + num_of_valid_sam > len(spaced_sentences):
        raise Exception("The number of tuning and validation samples together exceeds the total sample size of " + str(
            len(sentences)))
        
    sentence_shuffled = []
    summary_shuffled = []
    grammar_shuffled = []
    meaning_shuffled = []
    
    tune_shuffled = list(range(num_of_tuning_sam))
    random.shuffle(tune_shuffled)
    valid_shuffled = list(range(num_of_tuning_sam, num_of_tuning_sam + num_of_valid_sam))
    random.shuffle(valid_shuffled)
    train_shuffled = list(range(num_of_tuning_sam + num_of_valid_sam, len(spaced_sentences)))
    random.shuffle(train_shuffled)
    index_shuffled = tune_shuffled + valid_shuffled + train_shuffled

    for i in index_shuffled:
        sentence_shuffled.append(spaced_sentences[i])
        summary_shuffled.append(spaced_summaries[i])
        grammar_shuffled.append(tot_grammar[i])
        meaning_shuffled.append(tot_meaning[i])
    
    tuning_range = range(num_of_tuning_sam)
    valid_range = range(num_of_tuning_sam, num_of_tuning_sam + num_of_valid_sam)
    training_range = range(num_of_tuning_sam + num_of_valid_sam, len(summary_shuffled))

    output_for_grammar_files = [summary_shuffled, grammar_shuffled]
    __write_to_file(TUNE_FILE_PATH_GRAMMAR, tuning_range, output_for_grammar_files)
    __write_to_file(VALID_FILE_PATH_GRAMMAR, valid_range, output_for_grammar_files)
    __write_to_file(TRAIN_FILE_PATH_GRAMMAR, training_range, output_for_grammar_files)
    
    
    output_for_meaning_files =  [sentence_shuffled, summary_shuffled, meaning_shuffled]
    __write_to_file(TUNE_FILE_PATH_MEANING, tuning_range, output_for_meaning_files)
    __write_to_file(VALID_FILE_PATH_MEANING, valid_range, output_for_meaning_files)
    __write_to_file(TRAIN_FILE_PATH_MEANING, training_range, output_for_meaning_files)


def __write_to_file(output_path, index_range, lists):
    """ Write outputs to a tsv file.
    
    Args:
      output_path: the path of the output tsv file
      index_range: the range of indices in the list that will be written to the tsv file
      lists: a list of lists (columns) that will be written out to the tsv file
    """
    with open(os.path.expanduser(output_path), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for i in index_range:
            this_row = []
            for one_list in lists:
                this_row.append(one_list[i])
            tsv_writer.writerow(this_row)
    print("-------", len(index_range), "samples wrote to", output_path, "-------")

if __name__ == "__main__":
    """
    Preprocess the Microsoft text summarization dataset.
    
    Data needs to be downloaded from https://www.microsoft.com/en-us/download/details.aspx?id=54262 and the abosolute
    path to the dataset directory is provided as a command line argument.

    Dataset is split into training, tuning, and validation sets, with the number of samples in the tuning and validation
    set being specified in the command line argument. The three sets are saved in three separate tsv files, and all the
    preprocessed data are saved in another tsv file.
    
    usage: preprocess_MS_dataset_for_classifier.py [-h] raw_data_dir num_of_tuning num_of_validation

    positional arguments:
      raw_data_dir       Absolute path to the RawData directory in the MS dataset.
      num_of_tuning      Number of tuning samples
      num_of_validation  Number of validation samples
    
    optional arguments:
      -h, --help         show this help message and exit
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_dir", help="Absolute path to the RawData directory in the MS dataset.")
    parser.add_argument("num_of_tuning", help="Number of tuning samples", type=int)
    parser.add_argument("num_of_validation", help="Number of validation samples", type=int)
    arguments = parser.parse_args()
    main(arguments)
