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
"""Utility functions for running inference with a LaserTagger model."""

from __future__ import absolute_import, division, print_function

import tagging


class LaserTaggerPredictor(object):
  """Class for computing and realizing predictions with LaserTagger."""
  def __init__(self, tf_predictor, example_builder, label_map):
    """Initializes an instance of LaserTaggerPredictor.

    Args:
      tf_predictor: Loaded Tensorflow model.
      example_builder: BERT example builder.
      label_map: Mapping from tags to tag IDs.
    """
    self._predictor = tf_predictor
    self._example_builder = example_builder
    self._id_2_tag = {
        tag_id: tagging.Tag(tag)
        for tag, tag_id in label_map.items()
    }

  def predict(self, sources):
    """Returns realized prediction for given sources."""
    # Predict tag IDs.
    keys = ['input_ids', 'input_mask', 'segment_ids']
    inputs = []

    examples = []
    example = self._example_builder.build_bert_example(sources[0])
    if example is None:
      raise ValueError("Example couldn't be built.")
    inputs = {key: [example.features[key]] for key in keys}
    examples.append(example)

    for source in sources[1:]:
      example = self._example_builder.build_bert_example(source)
      examples.append(example)
      for key in keys:
        inputs[key].append(example.features[key])

    outputs = self._predictor(inputs)['pred']

    predictions = []
    for i, output in enumerate(outputs):
      predicted_ids = output.tolist()
      # Realize output.
      examples[i].features['labels'] = predicted_ids
      examples[i].features['labels_mask'] = \
        [0] + [1] * (len(predicted_ids) - 2) + [0]
      labels = [
          self._id_2_tag[label_id]
          for label_id in examples[i].get_token_labels()
      ]
      predictions.append(examples[i].editing_task.realize_output(labels))

    return predictions
