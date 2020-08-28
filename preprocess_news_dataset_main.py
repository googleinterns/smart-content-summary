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
"""Preprocess news summarization dataset. """

import argparse
import csv
import os

import pandas as pd

import preprocess_utils

PREPROCESSED_FILE_PATH = "~/preprocessed_news_dataset.tsv"
TRAIN_FILE_PATH = "~/train_news_dataset.tsv"
TUNE_FILE_PATH = "~/tune_news_dataset.tsv"
VALID_FILE_PATH = "~/valid_news_dataset.tsv"


def main(args):
  """Preprocess the news dataset.

    Args:
      args: command line arguments
    Raises:
      ValueError when dataset cannot be found in the path provided
    """
  num_of_tuning_sam = args.num_of_tuning
  num_of_valid_sam = args.num_of_validation

  if num_of_valid_sam < 0 or num_of_tuning_sam < 0:
    raise Exception("Number of samples must be non-negative integers")

  data_file_1 = args.news_summary_path
  data_file_2 = args.news_summary_more_path

  if not os.path.isfile(os.path.expanduser(PREPROCESSED_FILE_PATH)):
    if not os.path.isfile(os.path.expanduser(data_file_1)):
      raise ValueError(
          "Cannot find" + os.path.expanduser(data_file_1) +
          ". If necessary, please download from https://www.kaggle.com/sunnysai12345/news-summary"
      )

    if not os.path.isfile(os.path.expanduser(data_file_2)):
      raise ValueError(
          "Cannot find" + os.path.expanduser(data_file_2) +
          ". If necessary, please download from https://www.kaggle.com/sunnysai12345/news-summary"
      )

    dataset1 = (pd.read_csv(data_file_1,
                            encoding='iso-8859-1')).iloc[:, 0:6].copy()
    dataset2 = (pd.read_csv(data_file_2,
                            encoding='iso-8859-1')).iloc[:, 0:2].copy()

    dataset = pd.DataFrame()
    dataset['sentences'] = pd.concat([dataset1['text'], dataset2['text']],
                                     ignore_index=True)
    dataset['summaries'] = pd.concat(
        [dataset1['headlines'], dataset2['headlines']], ignore_index=True)

    cleaned_sentences = preprocess_utils.text_strip(dataset['sentences'])
    cleaned_summaries = preprocess_utils.text_strip(dataset['summaries'])

    cleaned_sentences, cleaned_summaries = preprocess_utils.delete_empty_entry(
        cleaned_sentences, cleaned_summaries)

    preprocess_utils.validate_dataset(cleaned_sentences, cleaned_summaries)
    print("Number of samples is", len(cleaned_sentences))

    preprocess_utils.calculate_stats(cleaned_sentences, cleaned_summaries)
    spaced_sentences = preprocess_utils.tokenize_with_space(cleaned_sentences)
    spaced_summaries = preprocess_utils.tokenize_with_space(cleaned_summaries)

    with open(os.path.expanduser(PREPROCESSED_FILE_PATH), 'wt') as out_file:
      tsv_writer = csv.writer(out_file, delimiter='\t')
      for i in range(len(spaced_sentences)):
        tsv_writer.writerow([spaced_sentences[i], spaced_summaries[i]])
    print("-------Preprocessed data saved to", PREPROCESSED_FILE_PATH,
          "-------")
  else:
    print("-------Preprocessed data exists. Now splitting dataset.-------")
  print("-------Now splitting dataset.-------")
  preprocess_utils.split_dataset(TRAIN_FILE_PATH,
                                 TUNE_FILE_PATH,
                                 VALID_FILE_PATH,
                                 PREPROCESSED_FILE_PATH,
                                 num_of_tuning_sam,
                                 num_of_valid_sam,
                                 whether_shuffle_entire_set=False,
                                 whether_shuffle_individual_file=True)


if __name__ == "__main__":
  """Preprocess the news summarization dataset.

    Data needs to be downloaded from https://www.kaggle.com/sunnysai12345/news-summary as two csv files news_summary.csv 
    and news_summary_more.csv. The absolute path to these two files are provided as command line arguments.

    Dataset is split into training, tuning, and validation sets, with the number of samples in the tuning and validation
    set being specified in the command line argument. The three sets are saved in three separate tsv files, and all the
    preprocessed data are saved in another tsv file.

    usage: preprocess_news_dataset.py [-h] news_summary_path news_summary_more_path num_of_tuning num_of_validation

    positional arguments:
      news_summary_path     Absolute path to the news_summary.csv
      news_summary_more_path
                            Absolute path to the news_summary_more.csv
      num_of_tuning         Number of tuning samples
      num_of_validation     Number of validation samples
    
    optional arguments:
      -h, --help            show this help message and exit
    """
  parser = argparse.ArgumentParser()
  parser.add_argument("news_summary_path",
                      help="Absolute path to the news_summary.csv")
  parser.add_argument("news_summary_more_path",
                      help="Absolute path to the news_summary_more.csv")
  parser.add_argument("num_of_tuning",
                      help="Number of tuning samples",
                      type=int)
  parser.add_argument("num_of_validation",
                      help="Number of validation samples",
                      type=int)
  arguments = parser.parse_args()
  main(arguments)
