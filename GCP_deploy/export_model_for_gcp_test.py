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
"""Testing for export_model_for_gcp.py."""

import shutil
import subprocess
import unittest
from unittest import TestCase

from export_model_for_gcp import re_export_model_LaserTagger
import tensorflow as tf

TEMP_TEST_FOLDER_1 = "./temp_folder_test_export_model_for_gcp_1"
TEMP_TEST_FOLDER_2 = "./temp_folder_test_export_model_for_gcp_2"


class ExportModelForGCPTest(TestCase):
  def tearDown(self):
    shutil.rmtree(TEMP_TEST_FOLDER_1)
    shutil.rmtree(TEMP_TEST_FOLDER_2)

  def test_re_export_model_LaserTagger(self):
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
      tensor_info_input_ids = tf.saved_model.utils.build_tensor_info(
          tf.placeholder(tf.int64, [None, None], name="Placeholder"))
      tensor_info_input_mask = tf.saved_model.utils.build_tensor_info(
          tf.placeholder(tf.int64, [None, None], name="Placeholder"))
      tensor_info_segment_ids = tf.saved_model.utils.build_tensor_info(
          tf.placeholder(tf.int64, [None, None], name="Placeholder"))
      tensor_info_outputs = tf.saved_model.utils.build_tensor_info(
          tf.placeholder(tf.int32, [None, None], name="loss/sub"))

    builder = tf.saved_model.builder.SavedModelBuilder(TEMP_TEST_FOLDER_1)
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
      prediction_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={
                  'input_ids': tensor_info_input_ids,
                  'input_mask': tensor_info_input_mask,
                  'segment_ids': tensor_info_segment_ids
              },
              outputs={'pred': tensor_info_outputs},
              method_name=tf.saved_model.signature_constants.
              PREDICT_METHOD_NAME))

      builder.add_meta_graph_and_variables(
          sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              prediction_signature
          })

      builder.add_meta_graph(
          [
              tf.saved_model.tag_constants.SERVING,
              tf.saved_model.tag_constants.TPU
          ],
          signature_def_map={
              tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              prediction_signature
          })
      builder.save()

    output = subprocess.check_output(("saved_model_cli show --dir=" +
                                      TEMP_TEST_FOLDER_1 + " --all").split())
    output = output.decode("utf-8").strip().split("\n")

    serve_metagraph = "MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:"
    serve_tpu_metagraph = "MetaGraphDef with tag-set: 'serve, tpu' contains the following SignatureDefs:"
    assert (serve_metagraph in output)
    assert (serve_tpu_metagraph in output)

    re_export_model_LaserTagger(TEMP_TEST_FOLDER_1, TEMP_TEST_FOLDER_2)
    output = subprocess.check_output(("saved_model_cli show --dir=" +
                                      TEMP_TEST_FOLDER_2 + " --all").split())
    output = output.decode("utf-8").strip().split("\n")
    self.assertTrue(serve_metagraph in output)
    self.assertTrue(serve_tpu_metagraph not in output)


if __name__ == '__main__':
  unittest.main()
