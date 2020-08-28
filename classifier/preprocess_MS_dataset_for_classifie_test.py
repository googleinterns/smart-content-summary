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

import unittest
from unittest import TestCase

from preprocess_MS_dataset_for_classifier import __find_most_common_bucket as find_most_common_bucket

TEMP_FILE_PATH = "./temp_test_mixing_in_negative_samples.tsv"


class PreprocessMSDatasetForClassifierTest(TestCase):
  def test_find_most_common_bucket(self):
    buckets = [["bucket1", "bucket2"], "bucket3"]
    items = ["bucket1", "bucket2", "bucket3"]

    output = find_most_common_bucket(items, buckets)
    self.assertEqual(output, 0)


if __name__ == '__main__':
  unittest.main()
