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
"""Testing for mixing_in_negative_samples.py."""

import csv
import os
import unittest
from unittest import TestCase

from mixing_in_negative_samples import calculate_number_of_synthetic_data_to_mix

TEMP_FILE_PATH = "./temp_test_mixing_in_negative_samples.tsv"


class MixingInNegativeSamplesTest(TestCase):
  def test_calculate_number_of_synthetic_data(self):
    with open(TEMP_FILE_PATH, "wt") as f:
      tsv_writer = csv.writer(f, delimiter='\t')
      for i in range(50):
        tsv_writer.writerow(["0"])
      for i in range(60):
        tsv_writer.writerow(["1"])
    output = calculate_number_of_synthetic_data_to_mix(TEMP_FILE_PATH, 0.8)
    os.remove(TEMP_FILE_PATH)
    self.assertEqual(output, 190)


if __name__ == '__main__':
  unittest.main()
