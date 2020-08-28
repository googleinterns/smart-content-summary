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

from keyword_clustering import cluster_keywords

TEMP_TESTING_FILE = "./temp_file_for_testing_keyword_clustering"


class KeywordClusteringTest(TestCase):
  def tearDown(self):
    os.remove(TEMP_TESTING_FILE)

  def test_keyword_clustering_with_nonempty_summary(self):
    with open(TEMP_TESTING_FILE, "wt") as f:
      tsv_writer = csv.writer(f, delimiter='\t')
      tsv_writer.writerow(["Original", "Model1", "Model2"])
      tsv_writer.writerow(["Object1", "Cluster1", "Cluster1"])
      tsv_writer.writerow(["Object2", "Cluster1", "Cluster1"])
      tsv_writer.writerow(["Object3", "Cluster1", "Cluster2"])
      tsv_writer.writerow(["Object4", "Cluster2", "Cluster2"])

    shortened_keywords_list, total_keyword_counts, model_name_list = cluster_keywords(
        TEMP_TESTING_FILE)
    self.assertEqual(shortened_keywords_list, [{
        'cluster1': ['Object1', 'Object2', 'Object3'],
        'cluster2': ['Object4']
    }, {
        'cluster1': ['Object1', 'Object2'],
        'cluster2': ['Object3', 'Object4']
    }])
    self.assertEqual(total_keyword_counts, 4)
    self.assertEqual(model_name_list, ["Model1", "Model2"])

  def test_keyword_clustering_with_empty_summary(self):
    with open(TEMP_TESTING_FILE, "wt") as f:
      tsv_writer = csv.writer(f, delimiter='\t')
      tsv_writer.writerow(["Original", "Model1", "Model2"])
      tsv_writer.writerow(["Object1", "", "Cluster1"])
      tsv_writer.writerow(["Object2", "Cluster1", "Cluster1"])
      tsv_writer.writerow(["Object3", "Cluster1", ""])
      tsv_writer.writerow(["Object4", "Cluster2", ""])

    shortened_keywords_list, total_keyword_counts, model_name_list = cluster_keywords(
        TEMP_TESTING_FILE)
    self.assertEqual(shortened_keywords_list, [{
        'cluster1': ['Object2', 'Object3'],
        'cluster2': ['Object4']
    }, {
        'cluster1': ['Object1', 'Object2']
    }])
    self.assertEqual(total_keyword_counts, 4)
    self.assertEqual(model_name_list, ["Model1", "Model2"])


if __name__ == '__main__':
  unittest.main()
