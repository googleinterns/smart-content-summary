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
# limitations under the License

"""Testing for predict_main.py."""

import csv 
import os
import subprocess
import unittest
from unittest import TestCase

TEMP_FILE_PATH = "./temp_file_for_testing.tsv"


class PredictMainTestCase(TestCase):    
    
    def test_predict_with_perfect_answer(self):
        with open(TEMP_FILE_PATH,"wt") as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerow(['This is a test source.', 'This is a test.', 'This is a test.'])
            
        output = subprocess.check_output(
            ("python score_main.py --prediction_file=" + TEMP_FILE_PATH).split())
        output = output.decode("utf-8").strip().split("\n")
        
        score_names = ["Exact score", "SARI score", "KEEP score", "ADDITION score", "DELETION score"]
        for i in range(len(output)):
            self.assertEqual(output[i].split(":")[0].strip(), score_names[i])
        
        for i in range(len(output)):
            self.assertEqual(float(output[i].split(":")[1]), 100)
        
        os.remove(TEMP_FILE_PATH)
        
    
    def test_predict_with_random_answer(self):
        with open(TEMP_FILE_PATH,"wt") as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerow(['This is a test source.', 'Random answer', 'This is a test.'])
            
        output = subprocess.check_output(
            ("python score_main.py --prediction_file=" + TEMP_FILE_PATH).split())
        output = output.decode("utf-8").strip().split("\n")
        
        self.assertEqual(float(output[0].split(":")[1]), 0)
        os.remove(TEMP_FILE_PATH)
    
    
    def test_sari_is_average_of_scores(self):
        with open(TEMP_FILE_PATH,"wt") as f:
            tsv_writer = csv.writer(f, delimiter='\t')
            tsv_writer.writerow(['This is a test source.', 'Random answer', 'This is a test.'])
            
        output = subprocess.check_output(
            ("python score_main.py --prediction_file=" + TEMP_FILE_PATH).split())
        output = output.decode("utf-8").strip().split("\n")
        
        sum_sari_subscores = 0
        for i in range(-3, 0):
            sum_sari_subscores += float(output[i].split(":")[1])
        
        self.assertAlmostEqual(sum_sari_subscores/3, 
                               float(output[-4].split(":")[1]), places=3)                       
        os.remove(TEMP_FILE_PATH)
        

if __name__ == '__main__':
    unittest.main()
