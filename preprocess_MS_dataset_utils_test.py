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
""" Testing for preprocess_MS_dataset_utils.py."""

import unittest
from unittest import TestCase

from preprocess_MS_dataset_utils import process_row


class PreprocessMSDatasetUtilsTest(TestCase):    
    def test_process_row_without_excluded_sample(self):
        row = ["PlaceHolder ||| PlaceHolder ||| OriginalSentence ||| " 
               "Summary1 ||| 6 ||| 6 ||| 6 ||| Most important meaning Flawless language "
               "||| Summary2 ||| 7 ||| 7 ||| 7 ||| Most important meaning Minor errors"]
        output_original_sentence, output_shortened_sentences_list, \
        output_shortened_ratings_list, count_excluded = process_row(row)
        
        self.assertEqual(output_original_sentence, 'OriginalSentence')
        self.assertEqual(output_shortened_sentences_list, ['Summary1', 'Summary2'])
        self.assertEqual(output_shortened_ratings_list, [['6'], ['7']])
        self.assertEqual(count_excluded, 0)
    
    
    def test_process_row_with_excluded_sample(self):
        row = ["PlaceHolder ||| PlaceHolder ||| OriginalSentence ||| " 
               "Summary1 ||| 7 ||| 7 ||| 7 ||| Most important meaning Minor errors "
               "||| Summary2 ||| 9 ||| 9 ||| 9 ||| Most important meaning Disfluent or incomprehensible"]
        output_original_sentence, output_shortened_sentences_list, \
        output_shortened_ratings_list, count_excluded = process_row(row)
        
        self.assertEqual(output_original_sentence, 'OriginalSentence')
        self.assertEqual(output_shortened_sentences_list, ['Summary1'])
        self.assertEqual(output_shortened_ratings_list, [['7']])
        self.assertEqual(count_excluded, 1)

        
if __name__ == '__main__':
    unittest.main()
