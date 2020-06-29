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
import sys

import tensorflow_datasets as tfds

import preprocess_utils

PREPROCESSED_FILE_PATH = "~/preprocessed_reddit_dataset.tsv"
TRAIN_FILE_PATH = "~/train_reddit_dataset.tsv"
TUNE_FILE_PATH = "~/tune_reddit_dataset.tsv"
VALID_FILE_PATH = "~/valid_reddit_dataset.tsv"


def main(argv):
    """ Preprocess the Reddit dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("num_of_tuning", help="Number of tuning samples", type=int)
    parser.add_argument("num_of_validation", help="Number of validation samples", type=int)

    args = parser.parse_args()
    num_of_tuning_sam = args.num_of_tuning
    num_of_valid_sam = args.num_of_validation

    if num_of_valid_sam < 0 or num_of_tuning_sam < 0:
        raise Exception("Number of samples must be non-negative integers")

    if not os.path.isfile(os.path.expanduser(PREPROCESSED_FILE_PATH)):
        ds = tfds.load('reddit_tifu', split='train', shuffle_files=True)

        sentences = []
        summaries = []
        for row in ds:
            summary = row["title"]
            sentence = row["tldr"]

            sentences.append(sentence.numpy().decode('UTF-8'))
            summaries.append(summary.numpy().decode('UTF-8'))

        cleaned_sentences = preprocess_utils.text_strip(sentences)
        cleaned_summaries = preprocess_utils.text_strip(summaries)

        cleaned_sentences, cleaned_summaries = preprocess_utils.delete_empty_entry(cleaned_sentences, cleaned_summaries)

        preprocess_utils.validate_dataset(cleaned_sentences, cleaned_summaries)
        print("Number of samples is", len(cleaned_sentences))

        preprocess_utils.calculate_stats(cleaned_sentences, cleaned_summaries)
        spaced_sentences = preprocess_utils.tokenize_with_space(cleaned_sentences)
        spaced_summaries = preprocess_utils.tokenize_with_space(cleaned_summaries)

        with open(os.path.expanduser(PREPROCESSED_FILE_PATH), 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for i in range(len(spaced_sentences)):
                tsv_writer.writerow([spaced_sentences[i], spaced_summaries[i]])
        print("-------Preprocessed data saved to", PREPROCESSED_FILE_PATH, "-------")
    else:
        print("-------Preprocessed data exists. Now splitting dataset.-------")
    print("-------Now splitting dataset.-------")
    preprocess_utils.split_dataset(TRAIN_FILE_PATH, TUNE_FILE_PATH, VALID_FILE_PATH, PREPROCESSED_FILE_PATH,
                                   num_of_tuning_sam, num_of_valid_sam, whether_shuffle_entire_set=False,
                                   whether_shuffle_individual_file=True)


if __name__ == "__main__":
    main(sys.argv[1:])
