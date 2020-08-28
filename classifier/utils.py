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
"""Utility functions for LaserTagger."""

from __future__ import absolute_import, division, print_function

import tensorflow as tf


def get_token_list(text):
  """Returns a list of tokens.

  This function expects that the tokens in the text are separated by space
  character(s). Example: "ca n't , touch". This is the case at least for the
  public DiscoFuse and WikiSplit datasets.

  Args:
    text: String to be split into tokens.
  """
  return text.split()


def yield_sources_and_targets_meaning(input_file):
  """Reads and yields source lists and targets from the input file.

  Args:
    input_file: Path to the input file.

  Yields:
    A tuple of (list of source texts, target text).
  """
  with tf.io.gfile.GFile(input_file) as f:
    for line in f:
      if len(line.rstrip('\n').split('\t')) == 3:
        source, summary, score = line.rstrip('\n').split('\t')
        yield [source], summary, score


def yield_sources_and_targets_grammar(input_file):
  """Reads and yields source lists and targets from the input file.

  Args:
    input_file: Path to the input file.

  Yields:
    A tuple of (list of source texts, target text).
  """
  with tf.io.gfile.GFile(input_file) as f:
    for line in f:
      source, score = line.rstrip('\n').split('\t')
      yield [source], None, score
