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
 
"""Testing for bert_example.py"""
    
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import bert_example
import tagging_converter
import tensorflow as tf

FLAGS = flags.FLAGS


class BertExampleTest(tf.test.TestCase):

  def setUp(self):
    super(BertExampleTest, self).setUp()

    vocab_tokens = ['[CLS]', '[SEP]', '[PAD]', 'a', 'b', 'c', '##d', '##e']
    vocab_file = os.path.join(FLAGS.test_tmpdir, 'vocab.txt')
    with tf.io.gfile.GFile(vocab_file, 'w') as vocab_writer:
      vocab_writer.write(''.join([x + '\n' for x in vocab_tokens]))

    max_seq_length = 8
    do_lower_case = False
    converter = tagging_converter.TaggingConverter([])
    self._builder = bert_example.BertExampleBuilder(
        vocab_file, max_seq_length, do_lower_case)
        

  def test_building_example_for_meaning(self):
    sources = "a b c"
    summary = "a b"
    label = 0
    example = self._builder.build_bert_example_meaning(sources, summary, label)
    self.assertEqual(example.features['input_ids_source'], [0, 3, 4, 5, 1, 2, 2, 2])
    self.assertEqual(example.features['input_mask_source'], [1, 1, 1, 1, 1, 0, 0, 0])
    self.assertEqual(example.features['segment_ids_source'], [0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(example.features['input_ids_summary'], [0, 3, 4, 1, 2, 2, 2, 2])
    self.assertEqual(example.features['input_mask_summary'], [1, 1, 1, 1, 0, 0, 0, 0])
    self.assertEqual(example.features['segment_ids_summary'], [0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(example.features['labels'], [label])


  def test_invaid_meaning_example(self):
    with self.assertRaises(ValueError):
        # The first feature list has len 2, whereas the others have len 1, so a
        # ValueError should be raised.
        bert_example.BertExampleMeaning([0] * 2, [0], [0], [0], [0], [0], [0], [0], [0])
  

  def test_building_example_for_grammar(self):
    text = "a b c"
    label = 0
    example = self._builder.build_bert_example_grammar(text, label)
    
    self.assertEqual(example.features['input_ids'], [0, 3, 4, 5, 1, 2, 2, 2])
    self.assertEqual(example.features['input_mask'], [1, 1, 1, 1, 1, 0, 0, 0])
    self.assertEqual(example.features['segment_ids'], [0, 0, 0, 0, 0, 0, 0, 0])
    self.assertEqual(example.features['labels'], [label])
    
  
  def test_invaid_grammar_example(self):
    with self.assertRaises(ValueError):
        # The first feature list has len 2, whereas the others have len 1, so a
        # ValueError should be raised.
        bert_example.BertExampleGrammar([0] * 2, [0], [0], [0], [0])


if __name__ == '__main__':
  tf.test.main()
