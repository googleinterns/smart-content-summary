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

# Lint as: python3
"""Utilities for building a LaserTagger TF model."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
from typing import Any, Mapping, Optional, Text
from bert import modeling
from bert import optimization
import transformer_decoder
import tensorflow as tf
from official_transformer import model_params


class LaserTaggerConfig(modeling.BertConfig):
  """Model configuration for LaserTagger."""

  def __init__(self, **kwargs):
    """Initializes an instance of LaserTagger configuration.

    This initializer expects BERT specific arguments.
    """
    super(LaserTaggerConfig, self).__init__(**kwargs)


class ModelFnBuilder(object):
  """Class for building `model_fn` closure for TPUEstimator."""

  def __init__(self, config, num_categories,
               init_checkpoint,
               learning_rate, num_train_steps,
               num_warmup_steps, use_tpu,
               use_one_hot_embeddings, max_seq_length,
               classifier_type):
    """Initializes an instance of a LaserTagger model.

    Args:
      config: LaserTagger model configuration.
      num_tags: Number of different tags to be predicted.
      init_checkpoint: Path to a pretrained BERT checkpoint (optional).
      learning_rate: Learning rate.
      num_train_steps: Number of training steps.
      num_warmup_steps: Number of warmup steps.
      use_tpu: Whether to use TPU.
      use_one_hot_embeddings: Whether to use one-hot embeddings for word
        embeddings.
      max_seq_length: Maximum sequence length.
      classifier_type: Either Grammar or Meaning.
    """
    self._config = config
    self._num_categories = num_categories
    self._init_checkpoint = init_checkpoint
    self._learning_rate = learning_rate
    self._num_train_steps = num_train_steps
    self._num_warmup_steps = num_warmup_steps
    self._use_tpu = use_tpu
    self._use_one_hot_embeddings = use_one_hot_embeddings
    self._max_seq_length = max_seq_length
    self._classifier_type = classifier_type

  def _create_model(self, mode, input_ids_source, input_mask_source, 
                    segment_ids_source, input_ids_summary,
                    input_mask_summary, segment_ids_summary, labels):
    """Creates a LaserTagger model."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model_source = modeling.BertModel(
        config=self._config,
        is_training=is_training,
        input_ids=input_ids_source,
        input_mask=input_mask_source,
        token_type_ids=segment_ids_source,
        use_one_hot_embeddings=self._use_one_hot_embeddings)
    final_hidden_source = model_source.get_sequence_output()
    
    if self._classifier_type == "Meaning":
        model_summary = modeling.BertModel(
            config=self._config,
            is_training=is_training,
            input_ids=input_ids_summary,
            input_mask=input_mask_summary,
            token_type_ids=segment_ids_summary,
            use_one_hot_embeddings=self._use_one_hot_embeddings)
        final_hidden_summary = model_source.get_sequence_output()

        final_hidden = tf.concat([final_hidden_source, final_hidden_summary],
                                axis=1)
    else:
        final_hidden = final_hidden_source

    if is_training:
    # I.e., 0.1 dropout
        final_hidden = tf.nn.dropout(final_hidden, keep_prob=0.9)

    layer1_output = tf.layers.dense(
        final_hidden,
        1,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        name="layer1")
    
    if self._classifier_type == "Meaning":
        flattened_layer1_output = tf.reshape(layer1_output, [-1, self._max_seq_length*2])
    else:
        flattened_layer1_output = tf.reshape(layer1_output, [-1, self._max_seq_length])
    logits = tf.expand_dims(tf.layers.dense(
        flattened_layer1_output, 
        self._num_categories,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        name="layer2"), 1)

    with tf.variable_scope("loss"):
      loss = None
      per_example_loss = None
      if mode != tf.estimator.ModeKeys.PREDICT:
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        loss = tf.reduce_mean(per_example_loss)
        pred = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
      else:
          pred = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

      return (loss, per_example_loss, pred)

  def build(self):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
      """The `model_fn` for TPUEstimator."""

      tf.logging.info("*** Features ***")
      for name in sorted(features.keys()):
        tf.logging.info("  name = %s, shape = %s", name, features[name].shape)

      if self._classifier_type == "Meaning":
        input_ids_source = features["input_ids_source"]
        input_mask_source = features["input_mask_source"]
        segment_ids_source = features["segment_ids_source"]

        input_ids_summary = features["input_ids_summary"]
        input_mask_summary = features["input_mask_summary"]
        segment_ids_summary = features["segment_ids_summary"]
      elif self._classifier_type == "Grammar":
        input_ids_source = features["input_ids"]
        input_mask_source = features["input_mask"]
        segment_ids_source = features["segment_ids"]

        input_ids_summary = None
        input_mask_summary = None
        segment_ids_summary = None
      else:
        raise ValueError("Classification type must be Grammar or Meaning")
        
      labels = None
      if mode != tf.estimator.ModeKeys.PREDICT:
        labels = features["labels"]

      (total_loss, per_example_loss, predictions) = self._create_model(
          mode, input_ids_source, input_mask_source, segment_ids_source, 
          input_ids_summary, input_mask_summary, segment_ids_summary, labels)

      tvars = tf.trainable_variables()
      initialized_variable_names = {}
      scaffold_fn = None
      if self._init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                        self._init_checkpoint)
        if self._use_tpu:
          def tpu_scaffold():
            tf.train.init_from_checkpoint(self._init_checkpoint, assignment_map)
            return tf.train.Scaffold()

          scaffold_fn = tpu_scaffold
        else:
          tf.train.init_from_checkpoint(self._init_checkpoint, assignment_map)

      tf.logging.info("**** Trainable Variables ****")
      for var in tvars:
        tf.logging.info("Initializing the model from: %s",
                        self._init_checkpoint)
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

      output_spec = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimization.create_optimizer(
            total_loss, self._learning_rate, self._num_train_steps,
            self._num_warmup_steps, self._use_tpu)

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)

      elif mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(per_example_loss, labels, labels_mask, predictions):
          """Compute eval metrics."""
          accuracy = tf.cast(
              tf.reduce_all(
                  tf.logical_or(
                      tf.equal(labels, predictions),
                      ~tf.cast(labels_mask, tf.bool)),
                  axis=1), tf.float32)
          return {
              # This is equal to the Exact score if the final realization step
              # doesn't introduce errors.
              "sentence_level_acc": tf.metrics.mean(accuracy),
              "eval_loss": tf.metrics.mean(per_example_loss),
          }

        eval_metrics = (metric_fn,
                        [per_example_loss, labels, labels_mask, predictions])
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn)
      else:
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions={"pred": predictions},
            scaffold_fn=scaffold_fn)
      return output_spec

    return model_fn
  
