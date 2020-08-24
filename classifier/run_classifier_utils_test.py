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

# Lint as: python3

"""Testing for run_classifier_utils.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import run_classifier_utils

import tensorflow as tf


def _get_model_builder(classifier_type):
  """Returns a LaserTagger model_fn builder."""
  config_json = {
      "attention_probs_dropout_prob":0.1,
      "hidden_act":"gelu",
      "hidden_dropout_prob":0.1,
      "hidden_size":768,
      "initializer_range":0.02,
      "intermediate_size":3072,
      "max_position_embeddings":512,
      "num_attention_heads":12,
      "num_hidden_layers":12,
      "type_vocab_size":2,
      "vocab_size":28996,
  }
  config = run_classifier_utils.LaserTaggerConfig(**config_json)
  return run_classifier_utils.ModelFnBuilder(
      config=config,
      num_categories=2,
      init_checkpoint=None,
      learning_rate=1e-4,
      num_train_steps=10,
      num_warmup_steps=1,
      use_tpu=False,
      use_one_hot_embeddings=False,
      max_seq_length=128,
      classifier_type=classifier_type)


class RunCLassifierUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(RunCLassifierUtilsTest, self).setUp()
    self._features = {
        "input_ids": [[0, 2, 3, 1] * 32],
        "input_mask": [[1, 1, 1, 1] * 32],
        "segment_ids": [[0] * 128],
        "input_ids_source": [[0, 2, 3, 1] * 32],
        "input_mask_source": [[1, 1, 1, 1] * 32],
        "segment_ids_source": [[0] * 128],
        "input_ids_summary": [[0, 2, 3, 1] * 32],
        "input_mask_summary": [[1, 1, 1, 1] * 32],
        "segment_ids_summary": [[0] * 128],
        "labels": [[0]]
    }
    self._features = {k: tf.convert_to_tensor(v)
                      for (k, v) in self._features.items()}

  @parameterized.parameters("Grammar", "Meaning")
  def test_create_model(self, classifier_type):
    input_ids = tf.constant([[0, 2, 3, 1] * 32], dtype=tf.int64)
    input_mask = tf.constant([[1, 1, 1, 1] * 32], dtype=tf.int64)
    segment_ids = tf.constant([[0] * 128], dtype=tf.int64)
    labels = tf.constant([[0]], dtype=tf.int64)

    model_fn_builder = _get_model_builder(classifier_type)
    (loss, _, pred) = model_fn_builder._create_model(
        tf.estimator.ModeKeys.TRAIN, input_ids, input_mask, 
        segment_ids, input_ids, input_mask, segment_ids, labels)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      out = sess.run({"loss": loss, "pred": pred})

      self.assertEqual(out["loss"].shape, ())
      self.assertEqual(out["pred"].shape, labels.shape)

  @parameterized.parameters("Grammar", "Meaning")
  def test_model_fn_train(self, classifier_type):
    with self.session() as sess:
      model_fn_builder = _get_model_builder(classifier_type)
      model_fn = model_fn_builder.build()
      output_spec = model_fn(
          self._features,
          labels=None,
          mode=tf.estimator.ModeKeys.TRAIN,
          params=None)

      sess.run([tf.global_variables_initializer(),
                tf.local_variables_initializer()])
      loss = sess.run(output_spec.loss)
      self.assertAllEqual(loss.shape, [])


if __name__ == "__main__":
  tf.test.main()
