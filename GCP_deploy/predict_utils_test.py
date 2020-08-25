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

"""Testing for predict_utils.py."""

import bert_example
import tensorflow as tf
import os
import subprocess
import shutil
import tagging
import tagging_converter
import unittest
from unittest import TestCase
import utils

from predict_utils import construct_example


class PredictUtilsTest(TestCase):    
    def test_construct_example(self):
        vocab_file = "gs://bert_traning_yechen/trained_bert_uncased/bert_POS/vocab.txt"
        label_map_file = "gs://publicly_available_models_yechen/best_hypertuned_POS/label_map.txt"
        enable_masking = False
        do_lower_case = True
        embedding_type = "POS"
        label_map = utils.read_label_map(label_map_file)
        converter = tagging_converter.TaggingConverter(
            tagging_converter.get_phrase_vocabulary_from_label_map(label_map), True)
        id_2_tag = {tag_id: tagging.Tag(tag) for tag, tag_id in label_map.items()}
        builder = bert_example.BertExampleBuilder(label_map, vocab_file, 10, do_lower_case,
                                          converter, embedding_type, enable_masking)

        inputs, example = construct_example("This is a test", builder)
        self.assertEqual(inputs, {'input_ids': [2, 12, 1016, 6, 9, 6, 9, 10, 12, 3], 
                          'input_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                          'segment_ids': [2, 16, 14, 14, 32, 14, 32, 5, 14, 41]})
        
    
if __name__ == '__main__':
    unittest.main()

