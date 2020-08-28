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
"""Testing for custom_predict.py"""

import csv
import os
import unittest
from unittest import TestCase

from custom_predict import __preprocess_input as preprocess_input
import preprocess_utils

TEMP_TESTING_FILE = "./temp_file_for_testing_custom_predict.tsv"


class CustomPredictTest(TestCase):
  def tearDown(self):
    os.remove(TEMP_TESTING_FILE)

  def test_preprocess_input_with_scoring_and_two_rows(self):
    with open(TEMP_TESTING_FILE, "wt") as f:
      tsv_writer = csv.writer(f, delimiter='\t')
      for i in range(10):
        tsv_writer.writerow(["Sample" + str(i), "Summary" + str(i)])

    sentences, summaries = preprocess_input(input_file_path=TEMP_TESTING_FILE,
                                            whether_score=True)

    preprocess_utils.validate_dataset(sentences, summaries)
    for i in range(10):
      self.assertEqual(sentences[i], "Sample" + str(i))
      self.assertEqual(summaries[i], "Summary" + str(i))

  def test_preprocess_input_with_scoring_and_only_one_row(self):
    with open(TEMP_TESTING_FILE, "wt") as f:
      tsv_writer = csv.writer(f, delimiter='\t')
      for i in range(10):
        tsv_writer.writerow(["Sample" + str(i)])

    with self.assertRaises(Exception):
      sentences, summaries = preprocess_input(
          input_file_path=TEMP_TESTING_FILE, whether_score=True)

  def test_preprocess_input_without_scoring_and_only_one_rows(self):
    with open(TEMP_TESTING_FILE, "wt") as f:
      tsv_writer = csv.writer(f, delimiter='\t')
      for i in range(10):
        tsv_writer.writerow(["Sample" + str(i)])

    sentences, summaries = preprocess_input(input_file_path=TEMP_TESTING_FILE,
                                            whether_score=False)

    preprocess_utils.validate_dataset(sentences, summaries)
    for i in range(10):
      self.assertEqual(sentences[i], "Sample" + str(i))
      self.assertEqual(summaries[i], "Sample" + str(i))


if __name__ == '__main__':
  unittest.main()
