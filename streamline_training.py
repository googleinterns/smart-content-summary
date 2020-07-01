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

import argparse
import os
import socket
import subprocess

"""Train LaserTagger model and export to GCP bucket."""

GCP_BUCKET = "gs://trained_models_yechen/"


def __validate_folders(args):
    """ Validate that the output_dir does not exist if it is starting from preprocessing. Validate that the folder
     containing LaserTagger scripts and Bert scripts and pretrained BERT exists.

     Args:
         args: the command line arguments
     """
    if not args.train and not args.export:
        if os.path.isdir(output_dir):
            raise Exception(output_dir + " is an existing folder. Please either delete it or rename it.")

    if not os.path.isdir(lasertagger_dir):
        raise Exception("LaserTagger not found.")
    if not os.path.isdir(os.path.expanduser(lasertagger_dir + "/bert")):
        raise Exception("Bert not found inside the LaserTagger folder.")

    if not os.path.isdir(os.path.expanduser(bert_dir)):
        raise Exception("Pretrained Bert model not found.")


def __validate_files(file):
    """ Validate that a file exists.

    Args:
        file: absolute file path
    """
    if not os.path.isfile(os.path.expanduser(file)):
        raise Exception("File " + str(file) + " not found/")


def __set_parameters(args):
    """ Set global variables based on the args.

    Args:
        args: command line arguments
    """
    global vocab_size, train_batch_size, learning_rate, num_train_epochs, warmup_proportion
    vocab_size = str(args.vocab_size)
    train_batch_size = str(args.train_batch_size)
    learning_rate = str(args.learning_rate)
    num_train_epochs = str(args.num_train_epochs)
    warmup_proportion = str(args.warmup_proportion)

    global output_dir, bert_dir, lasertagger_dir
    output_dir = os.path.expanduser(args.model_output_dir)
    bert_dir = os.path.expanduser(args.abs_path_to_bert)
    lasertagger_dir = os.path.expanduser(args.abs_path_to_lasertagger)

    global training_file, tuning_file
    training_file = os.path.expanduser(args.training_file)
    tuning_file = os.path.expanduser(args.tuning_file)


def __preprocess(args):
    """ Preprocess training data.

    Args:
        args: command line arguments
    """
    print("------ Running vocab optimization ------")
    output_file = output_dir + "/label_map.txt"

    with open(os.path.expanduser(args.training_file), 'r') as f:
        num_training = sum(1 for _ in f)

    if args.max_input_examples:
        if args.max_input_examples <= num_training:
            max_input_examples = str(args.max_input_examples)
        else:
            raise Exception("max input example > training data count")
    else:
        max_input_examples = str(int(num_training))

    print("number of maximum input is", max_input_examples)
    print("number of vocab size is", vocab_size)

    subprocess.call(("python phrase_vocabulary_optimization.py --input_file=" + training_file +
                     " --input_format=wikisplit" +
                     " --vocabulary_size=" + vocab_size +
                     " --max_input_examples=" + max_input_examples +
                     " --output_file=" + output_file).split(),
                    cwd=lasertagger_dir)

    print("------ Running preprocessing ------")
    subprocess.call(("python preprocess_main.py --input_file=" + tuning_file +
                     " --input_format=wikisplit" +
                     " --output_tfrecord=" + output_dir + "/tune.tf_record" +
                     " --label_map_file=" + output_file +
                     " --vocab_file=" + bert_dir + "/vocab.txt" +
                     " --output_arbitrary_targets_for_infeasible_examples=true").split(),
                    cwd=lasertagger_dir)

    subprocess.call(("python preprocess_main.py --input_file=" + training_file +
                     " --input_format=wikisplit" +
                     " --output_tfrecord=" + output_dir + "/train.tf_record" +
                     " --label_map_file=" + output_file +
                     " --vocab_file=" + bert_dir + "/vocab.txt" +
                     " --output_arbitrary_targets_for_infeasible_examples=false").split(),
                    cwd=lasertagger_dir)


def __training(args):
    """ Train the LaserTagger model."""
    print("------ Start training ------")
    f = open(output_dir + "/train.tf_record.num_examples.txt", "r")
    num_train_examples = f.read()
    f = open(output_dir + "/tune.tf_record.num_examples.txt", "r")
    num_tune_examples = f.read()

    print("training batch size is", train_batch_size)
    print("learning rate is", learning_rate)
    print("number of training epochs is", num_train_epochs)
    print("warm up proportion is", warmup_proportion)

    training_command = "python run_lasertagger.py" + \
                       " --do_train=true" + \
                       " --do_eval=true" + \
                       " --save_checkpoints_steps=500" + \
                       " --num_train_examples=" + num_train_examples + \
                       " --num_eval_examples=" + num_tune_examples + \
                       " --train_batch_size=" + train_batch_size + \
                       " --learning_rate=" + learning_rate + \
                       " --num_train_epochs=" + num_train_epochs + \
                       " --warmup_proportion" + warmup_proportion
    if args.use_tpu:
        print("Running on cloud TPU")
        bucket_name = args.gbucket
        tpu_name = socket.gethostname()
        folder_name = output_dir.split("/")[-1]
        folder_in_bucket = "gs://" + bucket_name + "/" + folder_name
        training_command += " --label_map_file=" + folder_in_bucket + "/label_map.txt" + \
                            " --model_config_file=" + folder_in_bucket + "/lasertagger_config.json" + \
                            " --init_checkpoint=" + folder_in_bucket + "/bert/bert_model.ckpt" + \
                            " --output_dir=" + folder_in_bucket + "/model" + \
                            " --training_file=" + folder_in_bucket + "/train.tf_record" + \
                            " --eval_file=" + folder_in_bucket + "/tune.tf_record" + \
                            " --use_tpu=true" + \
                            " --tpu_name=" + tpu_name
        subprocess.call(("gsutil -m cp -r " + output_dir + "/label_map.txt " +
                         folder_in_bucket + "/").split(), cwd=os.path.expanduser("~"))
        subprocess.call(("gsutil -m cp -r " + lasertagger_dir + "/configs/lasertagger_config.json " +
                         folder_in_bucket + "/").split(), cwd=os.path.expanduser("~"))
        subprocess.call(("gsutil -m cp -r " + bert_dir + "/* " +
                         folder_in_bucket + "/bert/").split(), cwd=os.path.expanduser("~"))
        subprocess.call(("gsutil -m cp -r " + output_dir + "/train.tf_record " +
                         folder_in_bucket + "/").split(), cwd=os.path.expanduser("~"))
        subprocess.call(("gsutil -m cp -r " + output_dir + "/tune.tf_record " +
                         folder_in_bucket + "/").split(), cwd=os.path.expanduser("~"))
    else:
        print("Running locally")
        training_command += " --label_map_file=" + output_dir + "/label_map.txt" + \
                            " --model_config_file=" + lasertagger_dir + "/configs/lasertagger_config.json" + \
                            " --init_checkpoint=" + bert_dir + "/bert_model.ckpt" + \
                            " --output_dir=" + output_dir + "/model" + \
                            " --training_file=" + output_dir + "/train.tf_record" + \
                            " --eval_file=" + output_dir + "/tune.tf_record"

    subprocess.call(training_command.split(), cwd=lasertagger_dir)

    if args.use_tpu:
        subprocess.call(("gsutil -m cp -r " + folder_in_bucket + "/model ./" + folder_name).split(),
                        cwd=os.path.expanduser("~"))

    print("------ Completed training ------")
    print("------ Start exporting ------")
    subprocess.call(("python run_lasertagger.py" +
                     " --label_map_file=" + output_dir + "/label_map.txt" +
                     " --model_config_file=" + lasertagger_dir + "/configs/lasertagger_config.json" +
                     " --output_dir=" + output_dir + "/model" +
                     " --do_export=true" +
                     " --export_path=" + output_dir + "/export").split(),
                    cwd=lasertagger_dir)
    os.rename(output_dir + "/export/" + os.listdir(output_dir + "/export")[0], output_dir + "/export/" + "export_model")


def __export_to_bucket():
    """ Export the LaserTagger model to the GCP bucket trained_models_yechen. """
    print("------ Exporting to bucket ------")
    folder_name = output_dir.split("/")[-1]
    subprocess.call(("gsutil -m cp -r " + output_dir + "/export/export_model " + GCP_BUCKET +
                     folder_name + "/").split(), cwd=os.path.expanduser("~"))
    subprocess.call(("gsutil -m cp -r " + output_dir + "/label_map.txt " + GCP_BUCKET + folder_name +
                     "/").split(), cwd=os.path.expanduser("~"))


if __name__ == "__main__":
    """ Streamline the preprocessing, training, and exporting process of the LaserTagger model using the hyperparameters
    in the command line arguments.
    
    usage: streamline_training.py [-h] [-vocab_size VOCAB_SIZE]
                              [-train_batch_size TRAIN_BATCH_SIZE]
                              [-learning_rate LEARNING_RATE]
                              [-num_train_epochs NUM_TRAIN_EPOCHS]
                              [-warmup_proportion WARMUP_PROPORTION]
                              [-max_input_examples MAX_INPUT_EXAMPLES]
                              [-train] [-export] [-use_tpu] [-gbucket GBUCKET]
                              model_output_dir abs_path_to_lasertagger
                              abs_path_to_bert training_file tuning_file
    
    positional arguments:
      model_output_dir      the directory of the model output
      abs_path_to_lasertagger
                            absolute path to the folder where the lasertagger
                            scripts are located
      abs_path_to_bert      absolute path to the folder where the pretrained BERT
                            is located
      training_file         path to training samples
      tuning_file           path to tuning samples
    
    optional arguments:
      -h, --help            show this help message and exit
      -vocab_size VOCAB_SIZE
                            vocab size. default = 500
      -train_batch_size TRAIN_BATCH_SIZE
                            batch size during training. default = 32
      -learning_rate LEARNING_RATE
                            The initial learning rate for Adam. default = 3e-5
      -num_train_epochs NUM_TRAIN_EPOCHS
                            Total number of training epochs to perform. default =
                            3
      -warmup_proportion WARMUP_PROPORTION
                            Proportion of training to perform linear learning rate
                            warmup for. default = 0.1
      -max_input_examples MAX_INPUT_EXAMPLES
                            number of training examples to use in the vocab
                            optimization. default is all training data.
      -train                If added, skip preprocessing and start training
      -export               If added, skip preprocessing and training, and start
                            exporting to bucket
      -use_tpu              If added, will use tpu for training
      -gbucket GBUCKET      The gcp bucket where cloud TPU will store intermediary
                            outputs to
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model_output_dir", help="the directory of the model output")
    parser.add_argument("abs_path_to_lasertagger",
                        help="absolute path to the folder where the lasertagger scripts are located")
    parser.add_argument("abs_path_to_bert", help="absolute path to the folder where the pretrained BERT is located")
    parser.add_argument("training_file", help="path to training samples")
    parser.add_argument("tuning_file", help="path to tuning samples")

    parser.add_argument("-vocab_size", type=int, help="vocab size. default = 500", default=500)
    parser.add_argument("-train_batch_size", type=int, help="batch size during training. default = 32", default=32)
    parser.add_argument("-learning_rate", type=float, help="The initial learning rate for Adam. default = 3e-5", default=3e-5)
    parser.add_argument("-num_train_epochs", type=int, help="Total number of training epochs to perform. default = 3", default=3)
    parser.add_argument("-warmup_proportion", type=float,
                        help="Proportion of training to perform linear learning rate warmup for. default = 0.1", default=0.1)
    parser.add_argument("-max_input_examples", type=int,
                        help="number of training examples to use in the vocab optimization. "
                             "default is all training data.")
    parser.add_argument("-train", action="store_true", help="If added, skip preprocessing and start training")
    parser.add_argument("-export", action="store_true",
                        help="If added, skip preprocessing and training, and start exporting to bucket")

    parser.add_argument("-use_tpu", action="store_true", help="If added, will use tpu for training")
    parser.add_argument("-gbucket", help="The gcp bucket where cloud TPU will store intermediary outputs to")

    args = parser.parse_args()

    if args.use_tpu and (args.gbucket is None):
        parser.error("-use_tpu requires -gbucket.")

    __set_parameters(args)
    __validate_folders(args)
    __validate_files(args.training_file)
    __validate_files(args.tuning_file)

    if args.export:
        print("------ Skipped preprocessing and training ------")
        __export_to_bucket()
    elif args.train:
        print("------ Skipped preprocessing ------")
        __training(args)
        __export_to_bucket()
    else:
        subprocess.call(['mkdir', os.path.expanduser(args.model_output_dir)], cwd=os.path.expanduser('~'))
        print("------Made new directory", args.model_output_dir, "------")
        __preprocess(args)
        __training(args)
        __export_to_bucket()
