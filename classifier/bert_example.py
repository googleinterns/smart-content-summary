# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build BERT Examples from text (source, target) pairs."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections

from bert import tokenization
import tagging
import tensorflow as tf
from typing import Mapping, MutableSequence, Optional, Sequence, Text


def to_tf_example_helper(bert_example):
    """Returns this object as a tf.Example."""

    def int_feature(values):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

    tf_features = collections.OrderedDict([
        (key, int_feature(val)) for key, val in bert_example.features.items()
    ])
    return tf.train.Example(features=tf.train.Features(feature=tf_features))


class BertExampleMeaning(object):
  """Class for training and inference examples for BERT.

  Attributes:
    features: Feature dictionary.
  """

  def __init__(self, input_ids_source,
               input_mask_source,
               segment_ids_source, 
               token_start_indices_source,
               input_ids_summary,
               input_mask_summary,
               segment_ids_summary, 
               token_start_indices_summary,
               labels):
        
    source_len = len(input_ids_source)
    if not (source_len == len(input_mask_source) and 
            source_len == len(segment_ids_source)):
      raise ValueError(
          'All feature lists should have the same length ({})'.format(
              source_len))
    
    summary_len = len(input_ids_summary)
    if not (summary_len == len(input_mask_summary) and 
            summary_len == len(segment_ids_summary)):
      raise ValueError(
          'All feature lists should have the same length ({})'.format(
              source_len))

    self.features = collections.OrderedDict([
        ('input_ids_source', input_ids_source),
        ('input_mask_source', input_mask_source),
        ('segment_ids_source', segment_ids_source),
        ('input_ids_summary', input_ids_summary),
        ('input_mask_summary', input_mask_summary),
        ('segment_ids_summary', segment_ids_summary),
        ('labels', labels)
    ])
    self._token_start_indices_source = token_start_indices_source
    self._token_start_indices_summary = token_start_indices_summary

  def pad_to_max_length(self, max_seq_length, pad_token_id):
    """Pad the feature vectors so that they all have max_seq_length.

    Args:
      max_seq_length: The length that features will have after padding.
      pad_token_id: input_ids feature is padded with this ID, other features
        with ID 0.
    """
    pad_len_source = max_seq_length - len(self.features['input_ids_source'])
    for key in ['input_ids_source', 'input_mask_source', 'segment_ids_source']:
      pad_id = pad_token_id if key == 'input_ids_source' else 0
      self.features[key].extend([pad_id] * pad_len_source)
    
    pad_len_summary = max_seq_length - len(self.features['input_ids_summary'])
    for key in ['input_ids_summary', 'input_mask_summary', 'segment_ids_summary']:
      pad_id = pad_token_id if key == 'input_ids_summary' else 0
      self.features[key].extend([pad_id] * pad_len_summary)
    
    for key in self.features:
      if key != 'labels' and len(self.features[key]) != max_seq_length:
        raise ValueError('{} has length {} (should be {}).'.format(
            key, len(self.features[key]), max_seq_length))

  def to_tf_example(self):
    """Returns this object as a tf.Example."""
    return to_tf_example_helper(self)


class BertExampleGrammar(object):
  """Class for training and inference examples for BERT.

  Attributes:
    features: Feature dictionary.
  """

  def __init__(self, input_ids,
               input_mask,
               segment_ids, 
               token_start_indices,
               labels):
        
    source_len = len(input_ids)
    if not (source_len == len(input_mask) and 
            source_len == len(segment_ids)):
      raise ValueError(
          'All feature lists should have the same length ({})'.format(
              source_len))

    self.features = collections.OrderedDict([
        ('input_ids', input_ids),
        ('input_mask', input_mask),
        ('segment_ids', segment_ids),
        ('labels', labels)
    ])
    self._token_start_indices = token_start_indices
    
  def pad_to_max_length(self, max_seq_length, pad_token_id):
    """Pad the feature vectors so that they all have max_seq_length.

    Args:
      max_seq_length: The length that features will have after padding.
      pad_token_id: input_ids feature is padded with this ID, other features
        with ID 0.
    """
    pad_len = max_seq_length - len(self.features['input_ids'])
    for key in ['input_ids', 'input_mask', 'segment_ids']:
      pad_id = pad_token_id if key == 'input_ids' else 0
      self.features[key].extend([pad_id] * pad_len)
      if len(self.features[key]) != max_seq_length:
        raise ValueError('{} has length {} (should be {}).'.format(
            key, len(self.features[key]), max_seq_length))
        
  def to_tf_example(self):
    """Returns this object as a tf.Example."""
    return to_tf_example_helper(self)


class BertExampleBuilder(object):
  """Builder class for BertExample objects."""

  def __init__(self, vocab_file,
               max_seq_length, do_lower_case):
    """Initializes an instance of BertExampleBuilder.

    Args:
      vocab_file: Path to BERT vocabulary file.
      max_seq_length: Maximum sequence length.
      do_lower_case: Whether to lower case the input text. Should be True for
        uncased models and False for cased models.
    """
    self._tokenizer = tokenization.FullTokenizer(vocab_file,
                                                 do_lower_case=do_lower_case)
    self._max_seq_length = max_seq_length
    self._pad_id = self._get_pad_id()

  def build_bert_example_meaning(self, sources, summaries, labels):
    """Constructs a BERT Example.

    Args:
      sources: List of source texts.
      summaries: List of corresponding summary texts.
      
    Returns:
      BertExample
    """
    input_ids_source, input_mask_source, segment_ids_source, \
    token_start_indices_source = self._get_embeddings(sources)
    
    input_ids_summary, input_mask_summary, segment_ids_summary, \
    token_start_indices_summary = self._get_embeddings(summaries)

    example = BertExampleMeaning(
        input_ids_source=input_ids_source,
        input_mask_source=input_mask_source,
        segment_ids_source=segment_ids_source,
        token_start_indices_source=token_start_indices_source,
        input_ids_summary=input_ids_summary,
        input_mask_summary=input_mask_summary,
        segment_ids_summary=segment_ids_summary,
        token_start_indices_summary=token_start_indices_summary,
        labels=[int(labels)]
    )
    example.pad_to_max_length(self._max_seq_length, self._pad_id)
    return example

  def build_bert_example_grammar(self, sources, labels):
    """Constructs a BERT Example.

    Args:
      sources: List of source texts.
      summaries: List of corresponding summary texts.
      
    Returns:
      BertExample
    """
    input_ids, input_mask, segment_ids, token_start_indices = \
      self._get_embeddings(sources)

    example = BertExampleGrammar(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        token_start_indices=token_start_indices,
        labels=[int(labels)]
    )
    example.pad_to_max_length(self._max_seq_length, self._pad_id)
    return example

  def _get_embeddings(self, text):
    """Get BERT embeddings for input text.
    
    Args:
      text: List of input texts.
    
    Returns:
      4-tuple of input_ids, input_mask, segment_ids, and 
      token_start_indices
    """
    tokens, token_start_indices = self._split_to_wordpieces(
        tagging.EditingTask(text).source_tokens)
    tokens = self._truncate_list(tokens)

    input_tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    
    return input_ids, input_mask, segment_ids, token_start_indices
        
  def _split_to_wordpieces(self, tokens):
    """Splits tokens (and the labels accordingly) to WordPieces.

    Args:
      tokens: Tokens to be split.
      labels: Labels (one per token) to be split.

    Returns:
      3-tuple with the split tokens, split labels, and the indices of the
      WordPieces that start a token.
    """
    bert_tokens = []  # Original tokens split into wordpieces.
    # Index of each wordpiece that starts a new token.
    token_start_indices = []
    for i, token in enumerate(tokens):
      # '+ 1' is because bert_tokens will be prepended by [CLS] token later.
      token_start_indices.append(len(bert_tokens) + 1)
      pieces = self._tokenizer.tokenize(token)
      bert_tokens.extend(pieces)
    return bert_tokens, token_start_indices

  def _truncate_list(self, x):
    """Returns truncated version of x according to the self._max_seq_length."""
    # Save two slots for the first [CLS] token and the last [SEP] token.
    return x[:self._max_seq_length - 2]

  def _get_pad_id(self):
    """Returns the ID of the [PAD] token (or 0 if it's not in the vocab)."""
    try:
      return self._tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
    except KeyError:
      return 0
