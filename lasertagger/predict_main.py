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
"""Compute realized predictions for a dataset."""

from __future__ import absolute_import, division, print_function

from absl import app, flags, logging
import bert_example
import predict_utils
import tagging_converter
import tensorflow as tf
import utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file', None,
    'Path to the input file containing examples for which to compute '
    'predictions.')
flags.DEFINE_enum('input_format', None, ['wikisplit', 'discofuse'],
                  'Format which indicates how to parse the input_file.')
flags.DEFINE_string(
    'output_file', None,
    'Path to the TSV file where the predictions are written to.')
flags.DEFINE_string(
    'label_map_file', None,
    'Path to the label map file. Either a JSON file ending with ".json", that '
    'maps each possible tag to an ID, or a text file that has one tag per '
    'line.')
flags.DEFINE_string('vocab_file', None, 'Path to the BERT vocabulary file.')
flags.DEFINE_integer('max_seq_length', 128, 'Maximum sequence length.')
flags.DEFINE_bool(
    'do_lower_case', False,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_bool('enable_swap_tag', True, 'Whether to enable the SWAP tag.')
flags.DEFINE_string('saved_model', None, 'Path to an exported TF model.')
flags.DEFINE_string(
    'embedding_type', None, 'Types of segment id embedding. If '
    'set to Normal, segment id is all zero. If set to Sentence, '
    'segment id marks sentences, i.e. 0 for first sentence, 1 for '
    'second second, 0 for third sentence, etc. If set to POS, '
    'segment id is the Part of Speech tag of each token. '
    'If set to POS_concise, segment is is the Part of Speech tags, '
    'but the number of tags is smaller than the one for POS embeddings.')
flags.DEFINE_bool('enable_masking', False, 'Whether to set digits and symbols'
                  'to generic type.')
flags.DEFINE_integer('batch_size', 1, 'Batch size of prediction.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  flags.mark_flag_as_required('input_file')
  flags.mark_flag_as_required('input_format')
  flags.mark_flag_as_required('output_file')
  flags.mark_flag_as_required('label_map_file')
  flags.mark_flag_as_required('vocab_file')
  flags.mark_flag_as_required('saved_model')
  flags.mark_flag_as_required('embedding_type')

  if FLAGS.batch_size < 0:
    raise ValueError("batch_size needs to be >= 1.")

  if FLAGS.batch_size == 1:
    logging.info(
        f'The prediction batch size is 1. Recommend a bigger batch size.')
  else:
    logging.info(f'The prediction batch size is {FLAGS.batch_size}.')

  label_map = utils.read_label_map(FLAGS.label_map_file)
  converter = tagging_converter.TaggingConverter(
      tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
      FLAGS.enable_swap_tag)
  builder = bert_example.BertExampleBuilder(label_map, FLAGS.vocab_file,
                                            FLAGS.max_seq_length,
                                            FLAGS.do_lower_case, converter,
                                            FLAGS.embedding_type,
                                            FLAGS.enable_masking)
  predictor = predict_utils.LaserTaggerPredictor(
      tf.contrib.predictor.from_saved_model(FLAGS.saved_model), builder,
      label_map)

  num_predicted = 0

  input_generator = utils.yield_sources_and_targets(FLAGS.input_file,
                                                    FLAGS.input_format)
  all_processed = False
  num_predicted = 0

  logging.info("----- Start prediction -----")
  with tf.gfile.Open(FLAGS.output_file, 'w') as writer:
    while not all_processed:
      batch_sources = []
      batch_targets = []
      for i in range(FLAGS.batch_size):
        try:
          source, target = next(input_generator)
          batch_sources.append(source)
          batch_targets.append(target)
        except StopIteration:
          all_processed = True

      if len(batch_targets) == 0:
        break

      predictions = predictor.predict(batch_sources)
      num_predicted += len(predictions)
      logging.log_every_n(logging.INFO, f'{num_predicted} predictions made.',
                          max(1, int(100 / FLAGS.batch_size)))
      for i, prediction in enumerate(predictions):
        writer.write(
            f'{" ".join(batch_sources[i])}\t{prediction}\t{batch_targets[i]}\n'
        )

  logging.info(f'{num_predicted} predictions saved to:\n{FLAGS.output_file}')


if __name__ == '__main__':
  app.run(main)
