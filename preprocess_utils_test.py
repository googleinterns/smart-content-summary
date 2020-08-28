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
""" Testing for preprocess_utils.py"""

import csv
from io import StringIO
import os
import subprocess
import unittest
from unittest import TestCase
from unittest.mock import patch

import preprocess_utils


class PreprocessUtilsTestCase(TestCase):
  def test_text_strip(self):
    # test remove escape characters
    test_case = ["\t", "\r", "\n", "\t" * 3, "\r" * 3, "\n" * 7]
    correct_ans = [" "] * 6
    result = preprocess_utils.text_strip(test_case)
    self.assertEqual(result, correct_ans)

    # test removing redundant special characters
    special_chars = ["_", "+", "-", "~", ":", "."]
    test_case = []
    correct_ans = []
    for char in special_chars:
      test_case.append("text" + char + "text")
      test_case.append("text" + char * 2 + "text")
      correct_ans += (["text" + char + "text"]) * 2
    result = preprocess_utils.text_strip(test_case)
    self.assertEqual(result, correct_ans)

    # test removing -, :, and _ at end or beginning of string (not in the middle)
    special_chars = ["-", ":", "_"]
    test_case = []
    correct_ans = []
    for char in special_chars:
      test_case.append("text" + char + "text")
      test_case.append(char + "text")
      test_case.append("text" + char)
      correct_ans += (["text" + char + "text"])
      correct_ans += (["text"]) * 2
    result = preprocess_utils.text_strip(test_case)
    self.assertEqual(result, correct_ans)

    # test removing ~ at end of string (not in the middle or beginning)
    test_case = ["text~", "text ~ text", "text~text", "~text"]
    correct_ans = ['text', 'text ~ text', 'text~text', '~text']
    result = preprocess_utils.text_strip(test_case)
    self.assertEqual(result, correct_ans)

    # test removing . at beginning of string (not in the middle or end)
    test_case = ["text.", "text . text", "text.text", ".text"]
    correct_ans = ['text.', 'text . text', 'text.text', 'text']
    result = preprocess_utils.text_strip(test_case)
    self.assertEqual(result, correct_ans)

  def test_validate_dataset_unequal(self):
    sentences = ["text"] * 5
    summaries = ["text"] * 7
    with self.assertRaises(Exception):
      preprocess_utils.validate_dataset(sentences, summaries)

    sentences = ["text"] * 7
    summaries = ["text"] * 7
    try:
      preprocess_utils.validate_dataset(sentences, summaries)
    except:
      self.fail("validate_dataset raised Exception unexpectedly!")

  def test_validate_dataset_empty(self):
    sentences = [" "] * 5
    summaries = ["text"] * 5
    with self.assertRaises(Exception):
      preprocess_utils.validate_dataset(sentences, summaries)

    summaries = [" "] * 5
    sentences = ["text"] * 5
    with self.assertRaises(Exception):
      preprocess_utils.validate_dataset(sentences, summaries)

    summaries = ["text"] * 3 + ["   "] * 2
    sentences = ["text"] * 5
    with self.assertRaises(Exception):
      preprocess_utils.validate_dataset(sentences, summaries)

  def test_calculate_stats(self):
    output_format = "-------Calculating statistics-------\n" + \
                    "Average word count of original sentence is {:.2f} ( std: {:.2f} )\n" + \
                    "Max word count is {}\n" + \
                    "Min word count is {}\n" + \
                    "Average word count of shortened sentence is {:.2f} ( std: {:.2f} )\n" + \
                    "Max Length is {}\n" + \
                    "Min Length is {}\n" + \
                    "On average, there are {:.2f} sentences in each original text ( std: {:.2f} )\n" + \
                    "On average, there are {:.2f} words in each shortened sentence that are not in the original " \
                    "sentence. ( std: {:.2f} )\n" + \
                    "The average compression ratio is {:.2f} ( std: {:.2f} )\n"

    sentences = ["sentences"]
    summaries = ["sum"]

    expected_output = output_format.format(1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0,
                                           1, 0)
    with patch('sys.stdout', new=StringIO()) as func_output:
      preprocess_utils.calculate_stats(sentences, summaries)
      self.assertEqual(func_output.getvalue(), expected_output)

    sentences = ["sentences 1", "sentences number 2"]
    summaries = ["sum 1", "2"]

    expected_output = output_format.format(2.5, 0.5, 3, 2, 1.5, 0.5, 2, 1, 1,
                                           0, 0.5, 0.5, 0.67, 0.33)
    with patch('sys.stdout', new=StringIO()) as func_output:
      preprocess_utils.calculate_stats(sentences, summaries)
      self.assertEqual(func_output.getvalue(), expected_output)

    summaries = ["text"] * 3 + ["   "] * 2
    sentences = ["text"] * 5
    with self.assertRaises(Exception):
      preprocess_utils.calculate_stats(sentences, summaries)

  def test_tokenize_with_space(self):
    test_case = ["I've eaten.", "Have you?"]
    correct_ans = ["I 've eaten .", "Have you ?"]
    result = preprocess_utils.tokenize_with_space(test_case)
    self.assertEqual(result, correct_ans)

  def test_split_dataset(self):
    tmp_folder_name = "tmp_test_split_dataset"
    subprocess.call(['rm', '-rf', tmp_folder_name],
                    cwd=os.path.expanduser('~'))
    subprocess.call(['mkdir', tmp_folder_name], cwd=os.path.expanduser('~'))

    spaced_sentences = []
    spaced_summaries = []
    for i in range(5):
      spaced_sentences += ["text " + str(i)]
      spaced_summaries += ["sum " + str(i)]

    proprocessed_file_path = "~/" + tmp_folder_name + "/preprocessed.tsv"
    with open(os.path.expanduser(proprocessed_file_path), 'wt') as out_file:
      tsv_writer = csv.writer(out_file, delimiter='\t')
      for i in range(len(spaced_sentences)):
        tsv_writer.writerow([spaced_sentences[i], spaced_summaries[i]])

    train_file_path = "~/" + tmp_folder_name + "/train.tsv"
    tune_file_path = "~/" + tmp_folder_name + "/tune.tsv"
    valid_file_path = "~/" + tmp_folder_name + "/valid.tsv"
    num_of_tuning_sam = 1
    num_of_valid_sam = 2
    whether_shuffle = True
    preprocess_utils.split_dataset(train_file_path, tune_file_path,
                                   valid_file_path, proprocessed_file_path,
                                   num_of_tuning_sam, num_of_valid_sam,
                                   whether_shuffle)

    with open(os.path.expanduser(train_file_path), 'r') as f:
      self.assertEqual(sum(1 for _ in f),
                       5 - num_of_tuning_sam - num_of_valid_sam)

    with open(os.path.expanduser(tune_file_path), 'r') as f:
      self.assertEqual(sum(1 for _ in f), num_of_tuning_sam)

    with open(os.path.expanduser(valid_file_path), 'r') as f:
      self.assertEqual(sum(1 for _ in f), num_of_valid_sam)

    # test the case where num_of_tuning_sam + num_of_valid_sam > num_of_total_sam
    num_of_tuning_sam = 5
    num_of_valid_sam = 5
    with self.assertRaises(Exception):
      preprocess_utils.split_dataset(train_file_path, tune_file_path,
                                     valid_file_path, proprocessed_file_path,
                                     num_of_tuning_sam, num_of_valid_sam,
                                     whether_shuffle)

    # test shuffling
    num_of_tuning_sam = 3
    num_of_valid_sam = 1
    whether_shuffle = False
    preprocess_utils.split_dataset(train_file_path, tune_file_path,
                                   valid_file_path, proprocessed_file_path,
                                   num_of_tuning_sam, num_of_valid_sam,
                                   whether_shuffle)
    sentences = []
    summaries = []
    with open(os.path.expanduser(tune_file_path), 'r') as f:
      tsv_reader = csv.reader(f, delimiter='\t')
      for line in tsv_reader:
        sentences.append(line[0])
        summaries.append(line[1])
    self.assertEqual(set(sentences),
                     set(spaced_sentences[0:num_of_tuning_sam]))
    self.assertEqual(set(summaries),
                     set(spaced_summaries[0:num_of_tuning_sam]))
    for sentence in sentences:
      self.assertEqual(summaries[sentences.index(sentence)],
                       spaced_summaries[spaced_sentences.index(sentence)])

    sentences = []
    summaries = []
    with open(os.path.expanduser(valid_file_path), 'r') as f:
      tsv_reader = csv.reader(f, delimiter='\t')
      for line in tsv_reader:
        sentences.append(line[0])
        summaries.append(line[1])
    self.assertEqual(
        set(sentences),
        set(spaced_sentences[num_of_tuning_sam:num_of_valid_sam +
                             num_of_tuning_sam]))
    self.assertEqual(
        set(summaries),
        set(spaced_summaries[num_of_tuning_sam:num_of_valid_sam +
                             num_of_tuning_sam]))
    for sentence in sentences:
      self.assertEqual(summaries[sentences.index(sentence)],
                       spaced_summaries[spaced_sentences.index(sentence)])

    subprocess.call(['rm', '-rf', tmp_folder_name],
                    cwd=os.path.expanduser('~'))

  def test_delete_empty_entry(self):
    summaries = []
    sentences = []

    for i in range(3):
      summaries += ["text " + str(i)]
      sentences += ["sum " + str(i)]
    summaries += ["  "] * 2
    sentences = ["text"] * 5

    sentences_cleaned, summaries_cleaned = preprocess_utils.delete_empty_entry(
        sentences, summaries)

    self.assertEqual(len(sentences_cleaned), 3)
    self.assertEqual(len(summaries_cleaned), 3)

    try:
      preprocess_utils.validate_dataset(sentences_cleaned, summaries_cleaned)
    except:
      self.fail("validate_dataset raised Exception unexpectedly!")

    for sentence in sentences_cleaned:
      self.assertEqual(summaries[sentences.index(sentence)],
                       summaries_cleaned[sentences_cleaned.index(sentence)])


if __name__ == '__main__':
  unittest.main()
