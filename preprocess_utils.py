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
"""Utilities for preprocessing input texts for LaserTagger."""

import csv
import os
import random
import re
from typing import List, Tuple, Text

import nltk
import numpy as np

nltk.download('punkt')


def text_strip(input_list: List[Text]) -> List[Text]:
  """Remove redundant special characters and escape characters from input text.

    Args:
      input_list: list of input texts.
    Returns:
      cleaned_list: the cleaned version of the text input list.
    """
  cleaned_list = []
  for text in input_list:
    # remove escape characters
    text = re.sub("(\\t)", ' ', str(text))
    text = re.sub("(\\r)", ' ', str(text))
    text = re.sub("(\\n)", ' ', str(text))

    # remove redundant special characters
    text = re.sub("(__+)", '_', str(text))
    text = re.sub("(--+)", '-', str(text))
    text = re.sub("(~~+)", '~', str(text))
    text = re.sub("(\+\++)", '+', str(text))
    text = re.sub("(\.\.+)", '.', str(text))
    text = re.sub("(\:\:+)", ':', str(text))

    # remove - at end or beginning of string (not in the middle)
    text = re.sub("(\-\s*$)", '', str(text))
    text = re.sub("(^\s*\-)", '', str(text))
    # remove : at end or beginning of string (not in the middle)
    text = re.sub("(\:\s*$)", '', str(text))
    text = re.sub("(^\s*\:)", '', str(text))
    # remove _ at end or beginning of string (not in the middle)
    text = re.sub("(_\s*$)", '', str(text))
    text = re.sub("(^\s*_)", '', str(text))
    # remove ~ at end of string (not in the middle or beginning)
    text = re.sub("(~\s*$)", '', str(text))
    # remove . at beginning of string (not in the middle or end)
    text = re.sub("(^\s*\.)", '', str(text))

    # remove multiple spaces
    text = re.sub("(\s+)", ' ', str(text))

    cleaned_list.append(text)

  return cleaned_list


def validate_dataset(sentences: List[Text], summaries: List[Text]):
  """Validate that the dataset has same number of sentences and summaries, and that it does not contain empty entry.

    Args:
      sentences: a list of input texts.
      summaries: a list of summaries corresponding to sentences.
    """
  if len(sentences) != len(summaries):
    raise Exception(
        "The number of original sentences does not match the number of summaries."
    )

  for sentence in sentences:
    if sentence.split() == 0:
      raise Exception("Original sentences contains empty examples.")

  for sentence in summaries:
    if sentence.split() == 0:
      raise Exception("Summaries contains empty examples.")


def calculate_stats(sentences: List[Text], summaries: List[Text]):
  """Calculate relevant statistics for the input sentences and their corresponding summaries.
    
    Relevant statistics include: average, maximum, and minimum number of words in original sentences and their 
    summaries; average number of sentences in each input sentences and summaries; average number of words that are 
    in the summary but not in the corresponding original sentence; and average compression ratio.
    
    Args:
      sentences: a list of input sentences.
      summaries: a list of summaries corresponding to the sentences in the sentences list.
    """

  print("-------Calculating statistics-------")
  validate_dataset(sentences, summaries)

  # average, maximum, and minimum number of words in original sentences and their summaries
  count_words = []
  for sentence in sentences:
    count_words.append(len(sentence.split()))
  count_words = np.array(count_words)
  print("Average word count of original sentence is",
        "{:.2f}".format(np.mean(count_words)), "( std:",
        "{:.2f}".format(np.std(count_words)), ")")
  print("Max word count is", np.max(count_words))
  print("Min word count is", np.min(count_words))

  count_words = []
  for summary in summaries:
    count_words.append(len(summary.split()))
  count_words = np.array(count_words)
  print("Average word count of shortened sentence is",
        "{:.2f}".format(np.mean(count_words)), "( std:",
        "{:.2f}".format(np.std(count_words)), ")")
  print("Max Length is", np.max(count_words))
  print("Min Length is", np.min(count_words))

  # average number of sentences in each input sentences and summaries; average number of words that are in the summary
  # but not in the corresponding original sentence
  count_diff = []
  count_sentences = []
  for i in range(len(sentences)):
    tokens_sentence = nltk.word_tokenize(sentences[i])
    tokens_sentence = [x.lower() for x in tokens_sentence]
    tokens_headline = nltk.word_tokenize(summaries[i])
    tokens_headline = [x.lower() for x in tokens_headline]
    count_diff.append(len(list(set(tokens_headline) - set(tokens_sentence))))
    count_sentences.append(
        np.max([
            tokens_sentence.count(".") + tokens_sentence.count("!") +
            tokens_sentence.count("?"), 1
        ]))
  count_sentences = np.array(count_sentences)
  count_diff = np.array(count_diff)
  print("On average, there are", "{:.2f}".format(np.mean(count_sentences)),
        "sentences in each original text", "( std:",
        "{:.2f}".format(np.std(count_sentences)), ")")
  print(
      "On average, there are", "{:.2f}".format(np.mean(count_diff)),
      "words in each shortened sentence that are not in the original sentence.",
      "( std:", "{:.2f}".format(np.std(count_diff)), ")")

  # average compression ratio
  compression_ratio = []
  for i in range(len(sentences)):
    compression_ratio.append(
        len(summaries[i].split()) / len(sentences[i].split()))
  compression_ratio = np.array(compression_ratio)
  print("The average compression ratio is",
        "{:.2f}".format(np.mean(compression_ratio)), "( std:",
        "{:.2f}".format(np.std(compression_ratio)), ")")


def tokenize_with_space(sentences: List[Text]) -> List[Text]:
  """Tokenize the input text with spaces separating the tokens.

    Args:
      sentences: a list of input sentences.
    
    Returns:
      spaced_sentences: a list of tokenized texts with spaces separating the tokens.
    """
  spaced_sentences = []
  for sentence in sentences:
    tokens = nltk.word_tokenize(sentence)
    spaced_sentence = " ".join(tokens)
    spaced_sentences.append(spaced_sentence)

  return spaced_sentences


def split_dataset(train_path: Text, tune_path: Text, valid_path: Text,
                  preprocessed_path: Text, num_of_tuning_sam: int,
                  num_of_valid_sam: int, whether_shuffle_entire_set: bool,
                  whether_shuffle_individual_file: bool):
  """Split the dataset into training, tuning, and validation sets, and store each in a file.

    Training set is stored in the path specified by train_path.
    Tuning set is stored in the path specified by tune_path.
    Validation set is stored in the path specified by valid_path.

    Args:
      train_path: absolute path to store the training set.
      tune_path: absolute path to store the tuning set.
      valid_path: absolute path to store the validation set.
      preprocessed_path: absolute path where the preprocessed dataset is store.
      num_of_tuning_sam: number of tuning samples.
      num_of_valid_sam: number of validation samples.
      whether_shuffle_entire_set: whether the dataset needs to be shuffled before splitting.
      whether_shuffle_individual_file: whether the dataset needs to be shuffled after splitting.
    Raises:
      ValueError when the number of tuning + validation sample together exceeds the total sample size
    """
  sentences = []
  summaries = []
  with open(os.path.expanduser(preprocessed_path), 'r') as f:
    tsv_reader = csv.reader(f, delimiter='\t')
    for line in tsv_reader:
      sentences.append(line[0])
      summaries.append(line[1])

  validate_dataset(sentences, summaries)

  sentence_shuffled = []
  summary_shuffled = []

  if num_of_tuning_sam + num_of_valid_sam > len(sentences):
    raise Exception(
        "The number of tuning and validation samples together exceeds the total sample size of "
        + str(len(sentences)))

  if whether_shuffle_entire_set:
    print("-------Shuffling the entire dataset-------")
    sentence_shuffled = []
    summary_shuffled = []
    index_shuffled = list(range(len(sentences)))
    random.shuffle(index_shuffled)
  elif whether_shuffle_individual_file:
    print(
        "-------Shuffling tuning, training, and testing set separately-------")
    tune_shuffled = list(range(num_of_tuning_sam))
    random.shuffle(tune_shuffled)
    valid_shuffled = list(
        range(num_of_tuning_sam, num_of_tuning_sam + num_of_valid_sam))
    random.shuffle(valid_shuffled)
    train_shuffled = list(
        range(num_of_tuning_sam + num_of_valid_sam, len(sentences)))
    random.shuffle(train_shuffled)
    index_shuffled = tune_shuffled + valid_shuffled + train_shuffled
  else:
    index_shuffled = list(range(len(sentences)))

  for i in index_shuffled:
    sentence_shuffled.append(sentences[i])
    summary_shuffled.append(summaries[i])
  sentences = sentence_shuffled
  summaries = summary_shuffled

  with open(os.path.expanduser(tune_path), 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i in range(num_of_tuning_sam):
      tsv_writer.writerow([sentences[i], summaries[i]])
  print("-------", num_of_tuning_sam, "tuning samples wrote to", tune_path,
        "-------")

  with open(os.path.expanduser(valid_path), 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i in range(num_of_tuning_sam, num_of_tuning_sam + num_of_valid_sam):
      tsv_writer.writerow([sentences[i], summaries[i]])
  print("-------", num_of_valid_sam, "validation samples wrote to", valid_path,
        "-------")

  with open(os.path.expanduser(train_path), 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i in range(num_of_tuning_sam + num_of_valid_sam, len(sentences)):
      tsv_writer.writerow([sentences[i], summaries[i]])
  print("-------",
        len(sentences) - num_of_tuning_sam - num_of_valid_sam,
        "tuning samples wrote to", train_path, "-------")


def delete_empty_entry(sentences: List[Text],
                       summaries: List[Text]) -> Tuple[List[Text], List[Text]]:
  """ Delete empty entries from the dataset.

    Args:
      sentences: a list of input sentences.
      summaries: a list of summaries corresponding to the sentences in the sentences list.

    Returns:
      sentences: a list of input sentences with empty entries deleted
      summaries: a list of summaries with empty entries deleted
    """
  empty_index = []
  for i, sentence in enumerate(sentences):
    if len(sentence.split()) == 0:
      empty_index.append(i)
  for i, summary in enumerate(summaries):
    if len(summary.split()) == 0:
      empty_index.append(i)
  empty_index = sorted(list(set(empty_index)), reverse=True)
  for index in empty_index:
    del sentences[index]
    del summaries[index]

  print("-------Deleted", len(empty_index), "empty entries-------")
  return sentences, summaries
