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
"""Testing for preprocess_MS_dataset_for_classifier.py."""

import csv
import os
import subprocess
from subprocess import DEVNULL
import unittest
from unittest import TestCase

TEMP_COLA_FILE_PATH = "./temp_test_cola_dataset.tsv"
TEMP_MS_FILE_PATH = "./temp_test_grammar_dataset.tsv"
OUTPUT_FILE_PATH = os.path.expanduser(
    "~/classifier_mixed_training_set_grammar.tsv")
OUTPUT_PREPROCESSED_FILE_PATH = os.path.expanduser(
    "~/classifier_preprocessed_cola_dataset.tsv")


class PreprocessCOLADatasetForClassifierTest(TestCase):
  def _populate_cola_files(self, number_negative_sample,
                           number_positive_sample):
    with open(TEMP_COLA_FILE_PATH, "wt") as f:
      tsv_writer = csv.writer(f, delimiter='\t')
      for i in range(number_negative_sample):
        tsv_writer.writerow(["placeholder", "0", "placeholder", "sample"])
      for i in range(number_positive_sample):
        tsv_writer.writerow(["placeholder", "1", "placeholder", "sample"])

  def _populate_MS_files(self, number_negative_sample, number_positive_sample):
    with open(TEMP_MS_FILE_PATH, "wt") as f:
      tsv_writer = csv.writer(f, delimiter='\t')
      for i in range(number_negative_sample):
        tsv_writer.writerow(["sample", "0"])
      for i in range(number_positive_sample):
        tsv_writer.writerow(["sample", "1"])

  def tearDown(self):
    os.remove(TEMP_COLA_FILE_PATH)
    os.remove(TEMP_MS_FILE_PATH)
    if os.path.exists(OUTPUT_FILE_PATH):
      os.remove(OUTPUT_FILE_PATH)
    if os.path.exists(OUTPUT_PREPROCESSED_FILE_PATH):
      os.remove(OUTPUT_PREPROCESSED_FILE_PATH)

  def test_preprocess_with_valid_ratio(self):
    self._populate_MS_files(500, 0)
    self._populate_cola_files(600, 600)
    goal_percentage = 0.8

    subprocess.run([
        "python", "preprocess_cola_dataset_for_classifier.py",
        "-goal_percentage_of_neg_samples=" + str(goal_percentage),
        TEMP_COLA_FILE_PATH, TEMP_MS_FILE_PATH
    ],
                   stdout=DEVNULL,
                   stderr=DEVNULL)

    negative_samples = 0
    total_samples = 0
    with open(OUTPUT_FILE_PATH) as f:
      tsvreader = csv.reader(f, delimiter="\t")
      for line in tsvreader:
        total_samples += 1
        if line[1] == "0":
          negative_samples += 1

    self.assertAlmostEqual(negative_samples / total_samples,
                           goal_percentage,
                           places=2)

  def test_preprocess_with_a_ratio_too_large(self):
    self._populate_MS_files(0, 500)
    self._populate_cola_files(500, 500)
    goal_percentage = 0.8

    with self.assertRaises(subprocess.CalledProcessError):
      subprocess.run([
          "python", "preprocess_cola_dataset_for_classifier.py",
          "-goal_percentage_of_neg_samples=" + str(goal_percentage),
          TEMP_COLA_FILE_PATH, TEMP_MS_FILE_PATH
      ],
                     check=True,
                     stdout=DEVNULL,
                     stderr=DEVNULL)

  def test_preprocess_with_a_ratio_too_small(self):
    self._populate_MS_files(0, 500)
    self._populate_cola_files(500, 0)
    goal_percentage = 0.4

    with self.assertRaises(subprocess.CalledProcessError):
      subprocess.run([
          "python", "preprocess_cola_dataset_for_classifier.py",
          "-goal_percentage_of_neg_samples=" + str(goal_percentage),
          TEMP_COLA_FILE_PATH, TEMP_MS_FILE_PATH
      ],
                     check=True,
                     stdout=DEVNULL,
                     stderr=DEVNULL)


if __name__ == '__main__':
  unittest.main()
