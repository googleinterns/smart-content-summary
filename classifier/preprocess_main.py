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
"""Convert a dataset into the TFRecord format.

The resulting TFRecord file will be used when training a LaserTagger model.
"""

from __future__ import absolute_import, division, print_function

from typing import Text

from absl import app, flags, logging
import bert_example
import tensorflow as tf
import utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file', None,
    'Path to the input file containing examples to be converted to '
    'tf.Examples.')
flags.DEFINE_string('output_tfrecord', None,
                    'Path to the resulting TFRecord file.')
flags.DEFINE_string('vocab_file', None, 'Path to the BERT vocabulary file.')
flags.DEFINE_integer('max_seq_length', 128, 'Maximum sequence length.')
flags.DEFINE_bool(
    'do_lower_case', False,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_string('classifier_type', None, 'The type of classification. '
                    '["Grammar", "Meaning"]')


def _write_example_count(count: int) -> Text:
  """Saves the number of converted examples to a file.

  This count is used when determining the number of training steps.

  Args:
    count: The number of converted examples.

  Returns:
    The filename to which the count is saved.
  """
  count_fname = FLAGS.output_tfrecord + '.num_examples.txt'
  with tf.io.gfile.GFile(count_fname, 'w') as count_writer:
    count_writer.write(str(count))
  return count_fname


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  flags.mark_flag_as_required('input_file')
  flags.mark_flag_as_required('output_tfrecord')
  flags.mark_flag_as_required("classifier_type")

  num_converted = 0

  if FLAGS.classifier_type == "Grammar":
    yield_example_fn = utils.yield_sources_and_targets_grammar
  elif FLAGS.classifier_type == "Meaning":
    yield_example_fn = utils.yield_sources_and_targets_meaning
  else:
    raise ValueError("classifier_type must be either Grammar or Meaning")

  builder = bert_example.BertExampleBuilder(
      FLAGS.vocab_file,
      FLAGS.max_seq_length,
      FLAGS.do_lower_case,
  )

  with tf.io.TFRecordWriter(FLAGS.output_tfrecord) as writer:
    for i, (sources, target,
            rating) in enumerate(yield_example_fn(FLAGS.input_file)):
      logging.log_every_n(
          logging.INFO,
          f'{i} examples processed, {num_converted} converted to tf.Example.',
          10000)
      if FLAGS.classifier_type == "Grammar":
        example = builder.build_bert_example_grammar(sources, rating)
      else:
        example = builder.build_bert_example_meaning(sources, target, rating)
      if example is None:
        continue
      writer.write(example.to_tf_example().SerializeToString())
      num_converted += 1
  logging.info(f'Done. {num_converted} examples converted to tf.Example.')
  count_fname = _write_example_count(num_converted)
  logging.info(f'Wrote:\n{FLAGS.output_tfrecord}\n{count_fname}')


if __name__ == '__main__':
  app.run(main)
