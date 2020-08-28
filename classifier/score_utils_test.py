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
"""Testing for score_utils.py"""

import unittest
from unittest import TestCase

from score_utils import calculate_recall_precision

class ScoreUtilsTest(TestCase):    
    def test_with_all_correct_output(self):
        precision, recall, score_f1 = calculate_recall_precision(
            [0,1,0,1], [0,1,0,1],0)
        self.assertAlmostEqual(precision, 1)
        self.assertAlmostEqual(recall, 1)
        self.assertAlmostEqual(score_f1, 1)


    def test_with_all_incorrect_output(self):
        precision, recall, score_f1 = calculate_recall_precision(
            [0,1,0,1], [1,0,1,0],0)
        self.assertAlmostEqual(precision, 0)
        self.assertAlmostEqual(recall, 0)
        self.assertAlmostEqual(score_f1, 0)
        
        
    def test_with_random_output(self):
        precision, recall, score_f1 = calculate_recall_precision(
            [0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0], 0)
        self.assertAlmostEqual(precision, 1/3)
        self.assertAlmostEqual(recall, 1/3)
        self.assertAlmostEqual(score_f1, 1/3)
        
if __name__ == '__main__':
    unittest.main()
