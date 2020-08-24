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

"""Testing for transformer_decoder.py."""

import numpy as np
import tensorflow as tf
import unittest
from unittest import TestCase

import transformer_decoder
from official_transformer import model_params


class TransformerDecoderTestCase(TestCase):    
    
    def test_transformer_decoder_without_target(self):
        batch_size = 256
        input_length = 128
        hidden_size = 768
        
        tf.reset_default_graph()
        with tf.Session() as sess:
            inputs = tf.constant(np.zeros([batch_size, input_length]), dtype="float32")
            encoder_outputs = tf.constant(np.zeros([batch_size, input_length, hidden_size]), dtype="float32")

            decoder = transformer_decoder.TransformerDecoder(
                params=model_params.BASE_PARAMS, train=True)
            result = decoder.__call__(inputs, encoder_outputs)

            self.assertIn('outputs', result)
            self.assertIn('scores', result)
        
    def test_transformer_decoder_with_target(self):
        batch_size = 256
        input_length = 128
        hidden_size = 768
        target_length = 128
        
        tf.reset_default_graph()
        with tf.Session() as sess:
            inputs = tf.constant(np.zeros([batch_size, input_length]), dtype="float32")
            encoder_outputs = tf.constant(np.zeros([batch_size, input_length, hidden_size]), dtype="float32")
            targets = tf.constant(np.zeros([batch_size, target_length]), dtype="int32")
            decoder = transformer_decoder.TransformerDecoder(
                params=model_params.BASE_PARAMS, train=True)
            result = decoder.__call__(inputs, encoder_outputs, targets)

            result_dimensions = result.get_shape().as_list()
            self.assertEqual(result_dimensions, [batch_size, input_length, model_params.BASE_PARAMS['vocab_size']])
        

if __name__ == '__main__':
    unittest.main()
