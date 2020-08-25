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

""" Testing for custom_post_processing_utils.py."""

import unittest
from unittest import TestCase
from custom_post_processing_utils import post_processing

class CustomPostProcessingUtils(TestCase):    
    def test_with_leading_unpaired_punctuation_marks(self):
        input_text = "' \" Test"
        output = post_processing(input_text)
        self.assertEqual(output, 'Test')
    
    
    def test_with_leading_unpaired_punctuation_marks_and_other_marks(self):
        input_text = "' \" . . Test"
        output = post_processing(input_text)
        self.assertEqual(output, 'Test')
    
    
    def test_with_leading_paired_marks_and_paired_marks_in_middle(self):
        input_text = "[ ] . . Test [ ] Test"
        output = post_processing(input_text)
        self.assertEqual(output, 'Test [] Test')
    
    
    def test_with_paired_marks(self):
        input_text = "[ Test ]"
        output = post_processing(input_text)
        self.assertEqual(output, '[ Test ]')
        
    
    def test_with_redundant_marks(self):
        input_text = "Test . ? Test"
        output = post_processing(input_text)
        self.assertEqual(output, 'Test . Test')
        

if __name__ == '__main__':
    unittest.main()
