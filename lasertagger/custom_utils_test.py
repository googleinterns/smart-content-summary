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

"""Testing for custom_utils.py."""

import json
import os
import nltk
import subprocess

import unittest
from unittest import TestCase
import custom_utils

from typing import Dict

TEMP_FOLDER_PATH = "~/temp_test_lasertagger_custom_utils"


def create_temp_folder():
    subprocess.call(['rm', '-rf', os.path.expanduser(TEMP_FOLDER_PATH)], cwd=os.path.expanduser('~'))
    subprocess.call(['mkdir', os.path.expanduser(TEMP_FOLDER_PATH)], cwd=os.path.expanduser('~'))

    
def generate_correct_output(vocab_size: int, vocab_type: int, t2t: bool, number_of_layer: int, hidden_size: int, attention_heads: int, filter_size: int, full_attention: bool) -> Dict:
    correct_output = {
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
    return correct_output

def clean_up():
    subprocess.call(['rm', '-rf', os.path.expanduser(TEMP_FOLDER_PATH)], cwd=os.path.expanduser('~'))
    
    
class CustomUtilsTestCase(TestCase):
    def test_write_lasertagger_config_write_POS_config(self):
        create_temp_folder()
        
        vocab_size = 32000
        vocab_type = 42
        t2t = True
        number_of_layer = 1
        hidden_size = 200
        attention_heads = 4
        filter_size = 400
        full_attention = False
        
        correct_output = generate_correct_output(vocab_size, vocab_type, t2t, number_of_layer, hidden_size, attention_heads, filter_size, full_attention)
        custom_utils.write_lasertagger_config(TEMP_FOLDER_PATH, "POS", t2t, number_of_layer, hidden_size, attention_heads, filter_size, full_attention)
        
        output_file_path = "{}/bert_config.json".format(os.path.expanduser(TEMP_FOLDER_PATH))
        with open(output_file_path, "r") as f:
            output_dict = json.load(f)

        self.assertDictEqual(correct_output, output_dict)
        
        clean_up()

    def test_write_lasertagger_config_write_normal_config(self):
        create_temp_folder()
        
        vocab_size = 28996
        vocab_type = 2
        t2t = False
        number_of_layer = 2
        hidden_size = 100
        attention_heads = 3
        filter_size = 300
        full_attention = True
        
        correct_output = generate_correct_output(vocab_size, vocab_type, t2t, number_of_layer, hidden_size, attention_heads, filter_size, full_attention)
        custom_utils.write_lasertagger_config(TEMP_FOLDER_PATH, "Base", t2t, number_of_layer, hidden_size, attention_heads, filter_size, full_attention)
        
        output_file_path = "{}/bert_config.json".format(os.path.expanduser(TEMP_FOLDER_PATH))
        with open(output_file_path, "r") as f:
            output_dict = json.load(f)

        self.assertDictEqual(correct_output, output_dict)
        
        clean_up()
    
    def test_write_lasertagger_config_raise_ValueError(self):
        create_temp_folder()
        
        t2t = True
        number_of_layer = 1
        hidden_size = 200
        attention_heads = 4
        filter_size = 400
        full_attention = False
        
        with self.assertRaises(ValueError):
            custom_utils.write_lasertagger_config(TEMP_FOLDER_PATH, "Bas", t2t, number_of_layer, hidden_size, attention_heads, filter_size, full_attention)
        
        clean_up()
        
    def test_convert_to_POS(self):
        result = custom_utils.convert_to_POS(["This is a test string"])
        self.assertTrue(max(result) <= 40 and min(result) >= 3)
    
if __name__ == '__main__':
    unittest.main()