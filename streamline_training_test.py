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

"""Testing for streamline_training.py."""

import json
import os
import nltk
import subprocess

import unittest
from unittest import TestCase
import streamline_training

from typing import Dict, Text

TEMP_FOLDER_PATH = "~/temp_test_streamline_training"
BERT_BASE_VOCAB_TYPE_NUMBER = 2
BERT_POS_VOCAB_TYPE_NUMBER = 42

BERT_BASE_VOCAB_SIZE = 28996
BERT_POS_VOCAB_SIZE = 32000
    
class StreamlineTrainingTest(TestCase):
    def __clean_up(self):
        subprocess.call(['rm', '-rf', os.path.expanduser(TEMP_FOLDER_PATH)], cwd=os.path.expanduser('~'))
        
    def __create_temp_folder(self):
        subprocess.call(['rm', '-rf', os.path.expanduser(TEMP_FOLDER_PATH)], cwd=os.path.expanduser('~'))
        subprocess.call(['mkdir', os.path.expanduser(TEMP_FOLDER_PATH)], cwd=os.path.expanduser('~'))


    def __generate_expected_output(self, vocab_size: int, vocab_type: int, t2t: bool, number_of_layer: int, 
                                  hidden_size: int, attention_heads: int, filter_size: int, 
                                  full_attention: bool) -> Dict:
        expected_output = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            'max_position_embeddings': 512,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": vocab_type,
            "vocab_size": vocab_size,
            "use_t2t_decoder": t2t,
            "decoder_num_hidden_layers": number_of_layer,
            "decoder_hidden_size": hidden_size,
            "decoder_num_attention_heads": attention_heads,
            "decoder_filter_size": filter_size,
            "use_full_attention": full_attention
        }
        return expected_output    
    
    
    def test_write_lasertagger_config_write_POS_config(self):
        self.__create_temp_folder()
        
        vocab_size = BERT_POS_VOCAB_SIZE
        vocab_type = BERT_POS_VOCAB_TYPE_NUMBER
        t2t = True
        number_of_layer = 1
        hidden_size = 200
        attention_heads = 4
        filter_size = 400
        full_attention = False
        
        expected_output = self.__generate_expected_output(vocab_size, vocab_type, t2t, number_of_layer, 
                                                        hidden_size, attention_heads, filter_size, 
                                                        full_attention)
        streamline_training.export_lasertagger_config_to_json(TEMP_FOLDER_PATH, "POS", t2t, number_of_layer, 
                                                              hidden_size, attention_heads, filter_size, 
                                                              full_attention)
        
        output_file_path = "{}/bert_config.json".format(os.path.expanduser(TEMP_FOLDER_PATH))
        with open(output_file_path, "r") as f:
            output_dict = json.load(f)

        self.assertDictEqual(expected_output, output_dict)
        
        self.__clean_up()

        
    def test_write_lasertagger_config_write_normal_config(self):
        self.__create_temp_folder()
        
        vocab_size = BERT_BASE_VOCAB_SIZE
        vocab_type = BERT_BASE_VOCAB_TYPE_NUMBER
        t2t = False
        number_of_layer = 2
        hidden_size = 100
        attention_heads = 3
        filter_size = 300
        full_attention = True
        
        expected_output = self.__generate_expected_output(vocab_size, vocab_type, t2t, number_of_layer, 
                                                        hidden_size, attention_heads, filter_size, 
                                                        full_attention)
        streamline_training.export_lasertagger_config_to_json(TEMP_FOLDER_PATH, "Base", t2t, number_of_layer, 
                                                              hidden_size, attention_heads, filter_size, 
                                                              full_attention)
        
        output_file_path = "{}/bert_config.json".format(os.path.expanduser(TEMP_FOLDER_PATH))
        with open(output_file_path, "r") as f:
            output_dict = json.load(f)

        self.assertDictEqual(expected_output, output_dict)
        
        self.__clean_up()
    
    
    def test_write_lasertagger_config_raise_ValueError(self):
        self.__create_temp_folder()
        
        t2t = True
        number_of_layer = 1
        hidden_size = 200
        attention_heads = 4
        filter_size = 400
        full_attention = False
        
        with self.assertRaises(ValueError):
            streamline_training.export_lasertagger_config_to_json(TEMP_FOLDER_PATH, "Bas", t2t, number_of_layer, 
                                                                  hidden_size, attention_heads, filter_size, 
                                                                  full_attention)
        
        self.__clean_up()
        
        
if __name__ == '__main__':
    unittest.main()
