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

"""Re-export SavedModel for GCP deployment."""

import argparse
import tensorflow as tf
import os

def re_export_model_LaserTagger(saved_model_folder, output_folder):
    saved_model_folder = os.path.expanduser(saved_model_folder)
    output_folder = os.path.expanduser(output_folder)
    
    print(output_folder)
    builder = tf.saved_model.builder.SavedModelBuilder(output_folder)

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        tensor_info_input_ids = tf.saved_model.utils.build_tensor_info(tf.placeholder(tf.int64, [None, None], name="Placeholder"))
        tensor_info_input_mask = tf.saved_model.utils.build_tensor_info(tf.placeholder(tf.int64, [None, None], name="Placeholder"))
        tensor_info_segment_ids = tf.saved_model.utils.build_tensor_info(tf.placeholder(tf.int64, [None, None], name="Placeholder"))
        tensor_info_outputs = tf.saved_model.utils.build_tensor_info(tf.placeholder(tf.int32, [None, None], name="loss/sub"))

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'input_ids': tensor_info_input_ids, 'input_mask': tensor_info_input_mask, 
                 'segment_ids': tensor_info_segment_ids},
          outputs={'pred': tensor_info_outputs},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        tf.compat.v1.saved_model.loader.load(sess, ["serve"], saved_model_folder)
        builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              prediction_signature 
      },
      )
        builder.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("saved_model_folder", help="Absolute path to the folder containing SavedModel.")
    parser.add_argument("output_folder", help="Absolute path to the folder to store re-exported model.")
    args = parser.parse_args()
    
    re_export_model(args.saved_model_folder, args.output_folder)
