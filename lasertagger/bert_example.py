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
import tagging_converter
import tensorflow as tf
from typing import Mapping, MutableSequence, Optional, Sequence, Text

import nltk
import custom_utils

POS_START_TAG = 2
POS_END_TAG = 41
POS_CONCISE_END_TAG = 16

class BertExample(object):
  """Class for training and inference examples for BERT.

  Attributes:
    editing_task: The EditingTask from which this example was created. Needed
      when realizing labels predicted for this example.
    features: Feature dictionary.
  """

  def __init__(self, input_ids,
               input_mask,
               segment_ids, labels,
               labels_mask,
               token_start_indices,
               task, default_label, embedding_type):
    input_len = len(input_ids)
    
    if not (input_len == len(input_mask) and input_len == len(segment_ids) and
            input_len == len(labels) and input_len == len(labels_mask)):
      raise ValueError(
          'All feature lists should have the same length ({})'.format(
              input_len))

    self.features = collections.OrderedDict([
        ('input_ids', input_ids),
        ('input_mask', input_mask),
        ('segment_ids', segment_ids),
        ('labels', labels),
        ('labels_mask', labels_mask),
    ])
    self._token_start_indices = token_start_indices
    self.editing_task = task
    self._default_label = default_label
    self._embedding_type = embedding_type

  def pad_to_max_length(self, max_seq_length, pad_token_id):
    """Pad the feature vectors so that they all have max_seq_length.

    Args:
      max_seq_length: The length that features will have after padding.
      pad_token_id: input_ids feature is padded with this ID, other features
        with ID 0.
    """
    pad_len = max_seq_length - len(self.features['input_ids'])
    for key in self.features:
      pad_id = pad_token_id if key == 'input_ids' else 0
      self.features[key].extend([pad_id] * pad_len)
      if len(self.features[key]) != max_seq_length:
        raise ValueError('{} has length {} (should be {}).'.format(
            key, len(self.features[key]), max_seq_length))

  def to_tf_example(self):
    """Returns this object as a tf.Example."""

    def int_feature(values):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

    tf_features = collections.OrderedDict([
        (key, int_feature(val)) for key, val in self.features.items()
    ])
    return tf.train.Example(features=tf.train.Features(feature=tf_features))

  def get_token_labels(self):
    """Returns labels/tags for the original tokens, not for wordpieces."""
    labels = []
    for idx in self._token_start_indices:
      # For unmasked and untruncated tokens, use the label in the features, and
      # for the truncated tokens, use the default label.
      if (idx < len(self.features['labels']) and
          self.features['labels_mask'][idx]):
        labels.append(self.features['labels'][idx])
      else:
        labels.append(self._default_label)
    return labels


class BertExampleBuilder(object):
  """Builder class for BertExample objects."""

  def __init__(self, label_map, vocab_file,
               max_seq_length, lower_case,
               converter, embedding_type, enable_mask):
    """Initializes an instance of BertExampleBuilder.

    Args:
      label_map: Mapping from tags to tag IDs.
      vocab_file: Path to BERT vocabulary file.
      max_seq_length: Maximum sequence length.
      lower_case: Whether to lower case the input text. Should be True for
        uncased models and False for cased models.
      converter: Converter from text targets to tags.
      embedding_type: POS or POS_concise or Normal or Sentence.
      enable_mask: whether to mask numbers and symbols
    """
    self._label_map = label_map
    self._tokenizer = tokenization.FullTokenizer(vocab_file,
                                                 lower_case=lower_case,
                                                 enable_mask=enable_mask)
    self._max_seq_length = max_seq_length
    self._converter = converter
    self._pad_id = self._get_pad_id()
    self._keep_tag_id = self._label_map['KEEP']
    if embedding_type not in ["POS", "Normal", "Sentence", "POS_concise"]:
        raise ValueError("Embedding_type must be Normal, POS, POS_concise, or Sentence") 
    self._embedding_type = embedding_type

  def build_bert_example(
      self,
      sources,
      target = None,
      use_arbitrary_target_ids_for_infeasible_examples = False
  ):
    """Constructs a BERT Example.

    Args:
      sources: List of source texts.
      target: Target text or None when building an example during inference.
      use_arbitrary_target_ids_for_infeasible_examples: Whether to build an
        example with arbitrary target ids even if the target can't be obtained
        via tagging.

    Returns:
      BertExample, or None if the conversion from text to tags was infeasible
      and use_arbitrary_target_ids_for_infeasible_examples == False.
    """
    # Compute target labels.
    task = tagging.EditingTask(sources)
    if target is not None:
      tags = self._converter.compute_tags(task, target)
      if not tags:
        if use_arbitrary_target_ids_for_infeasible_examples:
          # Create a tag sequence [KEEP, DELETE, KEEP, DELETE, ...] which is
          # unlikely to be predicted by chance.
          tags = [tagging.Tag('KEEP') if i % 2 == 0 else tagging.Tag('DELETE')
                  for i, _ in enumerate(task.source_tokens)]
        else:
          return None
    else:
      # If target is not provided, we set all target labels to KEEP.
      tags = [tagging.Tag('KEEP') for _ in task.source_tokens]
    labels = [self._label_map[str(tag)] for tag in tags]

    tokens, labels, token_start_indices, special_tags = self._split_to_wordpieces(
        task.source_tokens, labels, self._embedding_type)

    tokens = self._truncate_list(tokens)
    labels = self._truncate_list(labels)
    if special_tags is not None:
        special_tags = self._truncate_list(special_tags)

    input_tokens = ['[CLS]'] + tokens + ['[SEP]']
    labels_mask = [0] + [1] * len(labels) + [0]
    labels = [0] + labels + [0]

    input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)
    if self._embedding_type == "Normal":
        segment_ids = [0] * len(input_ids)
    elif self._embedding_type == "POS":
        segment_ids = [POS_START_TAG] + special_tags + [POS_END_TAG]
    elif self._embedding_type == "POS_concise":
        segment_ids = [POS_START_TAG] + special_tags + [POS_CONCISE_END_TAG]
    elif self._embedding_type == "Sentence":
        segment_ids = [0] + special_tags + [0]
    else:
        raise ValueError("Embedding_type must be Normal, POS, POS_concise, or Sentence") 

    example = BertExample(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        labels=labels,
        labels_mask=labels_mask,
        token_start_indices=token_start_indices,
        task=task,
        default_label=self._keep_tag_id,
        embedding_type=self._embedding_type)
    example.pad_to_max_length(self._max_seq_length, self._pad_id)
    return example


  def _split_to_wordpieces(self, tokens, labels, embedding_type):
    """Splits tokens (and the labels accordingly) to WordPieces.

    Args:
      tokens: Tokens to be split.
      labels: Labels (one per token) to be split.
      embedding_type: Normal, POS, POS_consie, or Sentence

    Returns:
      4-tuple with the split tokens, split labels, the indices of the
      WordPieces that start a token, and special tags for each token 
      (POS tag if embedding_type is POS, None if embedding_type is 
      Normal, and sentence tag if embedding_type is Sentence)
    """
    bert_tokens = []  # Original tokens split into wordpieces.
    bert_labels = []  # Label for each wordpiece.
    # Index of each wordpiece that starts a new token.
    token_start_indices = []
    for i, token in enumerate(tokens):
      # '+ 1' is because bert_tokens will be prepended by [CLS] token later.
      token_start_indices.append(len(bert_tokens) + 1)
      pieces = self._tokenizer.tokenize(token)
      bert_tokens.extend(pieces)
      bert_labels.extend([labels[i]] * len(pieces))
    
    if embedding_type == "Normal":
        return bert_tokens, bert_labels, token_start_indices, None
    elif embedding_type in ["POS", "POS_concise"]:
        tokens_pos = custom_utils.convert_to_pos(tokens, pos_type=embedding_type)
        pos_tags = []
        for i, token in enumerate(tokens):
            pieces = self._tokenizer.tokenize(token)
            pos_tags.extend([tokens_pos[i]] * len(pieces))
        return bert_tokens, bert_labels, token_start_indices, pos_tags
    elif embedding_type == "Sentence":
        sentence_tags = []
        sentence_counter = 0
        for i, token in enumerate(tokens):
            pieces = self._tokenizer.tokenize(token)
            for piece in pieces:
                sentence_tags.extend([sentence_counter])
                if piece in [".", ",", ";", "!", "?", ":", 
                               "##.", "##,", "##;", "##!", "##?", "##:"]:
                    sentence_counter = 1 - sentence_counter
        return bert_tokens, bert_labels, token_start_indices, sentence_tags
    else:
        raise ValueError("Embedding_type must be Normal, POS, POS_concise, or Sentence")

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
