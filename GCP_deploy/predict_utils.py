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

""" Construct BERT example for LaserTagger and Grammar Checer """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def construct_example(sentence, example_builder):
    """ Construct BERT model.
    Args:
      sentence: sentence to be converted to BERT example.
      example builder: BERT example builder.
      
    Returns:
      inputs: a dict with all features of the input sentence
      example: bert_example object
    """
    keys = ['input_ids', 'input_mask', 'segment_ids']
    example = example_builder.build_bert_example(sentence)

    if example is None:
        raise ValueError("Example couldn't be built.")

    inputs = {key: example.features[key] for key in keys}
    return inputs, example
