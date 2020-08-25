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
import json

from typing import Dict, Text

"""Train LaserTagger model and export to GCP bucket."""

GCP_BUCKET = "gs://trained_models_yechen/"
BERT_TYPE_BASE = "Base"
BERT_TYPE_POS = "POS"
BERT_TYPE_POS_CONCISE = "POS_concise"

BERT_BASE_VOCAB_TYPE_NUMBER = 2
BERT_POS_VOCAB_TYPE_NUMBER = 42
BERT_POS_CONCISE_VOCAB_TYPE_NUMBER = 16

BERT_BASE_VOCAB_SIZE = 28996
BERT_POS_VOCAB_SIZE = 32000
BERT_POS_CONCISE_VOCAB_SIZE = 32000

def export_lasertagger_config_to_json(output_dir: Text, bert_type: Text, t2t: bool, number_of_layer: int, 
                                      hidden_size: int, attention_heads: int, filter_size: int, 
                                      full_attention: bool):
    """ Write the LaserTagger configuration as a json file.
    
    Args:
        file_dir: the directory where the json file will be stored
        bert_type: the type of Bert. There are three types: Base, POS, and POS_concise
        t2t: If True, will use autoregressive decoder. If False, will use feedforward decoder.
        number_of_layer: number of hidden layers in the decoder.
        hidden_size: the size of hidden layer in the decoder.
        attention_heads: number of attention heads in the decoder. 
        filter_size: the size of the decoder filter.
        full_attention: whether to use full attention in the decoder. 
    """
    
    if bert_type == BERT_TYPE_BASE:
        vocab_type = BERT_BASE_VOCAB_TYPE_NUMBER
        vocab_size = BERT_BASE_VOCAB_SIZE
    elif bert_type == BERT_TYPE_POS:
        vocab_type = BERT_POS_VOCAB_TYPE_NUMBER
        vocab_size = BERT_POS_VOCAB_SIZE
    elif bert_type == BERT_TYPE_POS_CONCISE:
        vocab_type = BERT_POS_CONCISE_VOCAB_TYPE_NUMBER
        vocab_size = BERT_POS_CONCISE_VOCAB_SIZE
    else:
        raise ValueError("bert_type needs to be 'Base', 'POS', or 'POS_concise'.")
        
    
    lasertagger_conf = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        'max_position_embeddings': 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": vocab_type,
        "vocab_size": vocab_size,
        "use_t2t_decoder": t2t,
        "decoder_num_hidden_layers": number_of_layer,
        "decoder_hidden_size": hidden_size,
        "decoder_num_attention_heads": attention_heads,
        "decoder_filter_size": filter_size,
        "use_full_attention": full_attention
    }
    
    output_dir = os.path.expanduser(output_dir)

    with open("{}/lasertagger_config.json".format(output_dir), "w") as f:
        json.dump(lasertagger_conf, f, indent=2)


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
    if args.embedding_type not in ["Normal", "POS", "Sentence", "POS_concise"]:
        raise ValueError("Embedding_type must be Normal, POS, POS_concise, or Sentence")
    
    if args.verb_deletion_loss != 0 and args.embedding_type not in ["POS", "POS_concise"]:
        raise ValueError("Verb deletion loss is non-zero and the embedding type is not POS.")
    
    if args.verb_deletion_loss < 0:
        raise ValueError("Verb deletion loss weight must be greater than 0.")
    
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
    tuning_preprocess_command = ("python preprocess_main.py --input_file=" + tuning_file + 
                     " --input_format=wikisplit" + 
                     " --output_tfrecord=" + output_dir + "/tune.tf_record" +
                     " --label_map_file=" + output_file +
                     " --vocab_file=" + bert_dir + "/vocab.txt" +
                     " --output_arbitrary_targets_for_infeasible_examples=true" +
                     " --embedding_type=" + args.embedding_type)
    if args.masking:
        tuning_preprocess_command += " --enable_mask=true"
                        
    subprocess.call(tuning_preprocess_command.split(),
                    cwd=lasertagger_dir)
    
    training_preprocess_command = ("python preprocess_main.py --input_file=" + training_file +
                     " --input_format=wikisplit" +
                     " --output_tfrecord=" + output_dir + "/train.tf_record" +
                     " --label_map_file=" + output_file +
                     " --vocab_file=" + bert_dir + "/vocab.txt" +
                     " --output_arbitrary_targets_for_infeasible_examples=false" +
                     " --embedding_type=" + args.embedding_type)
    if args.masking:
        training_preprocess_command += " --enable_mask=true"                    
    
    subprocess.call(training_preprocess_command.split(),
                    cwd=lasertagger_dir)


def __training(args):
    """ Train the LaserTagger model."""
    print("------ Generating config file ------")
    
    if args.embedding_type in ["Normal", "Sentence"]:
        bert_type = "Base"
    else:
        bert_type = args.embedding_type
    
    export_lasertagger_config_to_json(output_dir, bert_type, args.t2t, args.number_layer, args.hidden_size, 
                                      args.num_attention_head, args.filter_size, args.full_attention)
    config_file_path = "{}/lasertagger_config.json".format(output_dir)
    
    print("------ Start training ------")
    with open(output_dir + "/train.tf_record.num_examples.txt", "r") as f:
        num_train_examples = f.read()
    with open(output_dir + "/tune.tf_record.num_examples.txt", "r") as f:
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
                       " --warmup_proportion" + warmup_proportion + \
                       " --verb_loss_weight" + str(args.verb_deletion_loss) + \
                       " --embedding_type" + args.embedding_type + \
                       " --add_tag_loss_weight" + str(args.add_tag_loss_weight) + \
                       " --delete_tag_loss_weight" + str(args.delete_tag_loss_weight) + \
                       " --keep_tag_loss_weight" + str(args.keep_tag_loss_weight) 
    
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
        subprocess.call(("gsutil -m cp -r " + config_file_path + " " + 
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
                            " --model_config_file=" + config_file_path + \
                            " --init_checkpoint=" + bert_dir + "/bert_model.ckpt" + \
                            " --output_dir=" + output_dir + "/model" + \
                            " --training_file=" + output_dir + "/train.tf_record" + \
                            " --eval_file=" + output_dir + "/tune.tf_record"

    subprocess.call(training_command.split(), cwd=lasertagger_dir)

    if args.use_tpu:
        subprocess.call(("gsutil -m cp -r " + folder_in_bucket + "/model " + output_dir).split(),
                        cwd=os.path.expanduser("~"))

    print("------ Completed training ------")
    print("------ Start exporting ------")
    subprocess.call(("python run_lasertagger.py" +
                     " --label_map_file=" + output_dir + "/label_map.txt" +
                     " --model_config_file=" + config_file_path +
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
    
    usage: streamline_training.py [-h] [-vocab_size VOCAB_SIZE] [-train_batch_size TRAIN_BATCH_SIZE] 
        [-learning_rate LEARNING_RATE] [-num_train_epochs NUM_TRAIN_EPOCHS] [-warmup_proportion WARMUP_PROPORTION]
        [-max_input_examples MAX_INPUT_EXAMPLES] [-train] [-export] [-use_tpu] [-gbucket GBUCKET] [-t2t T2T] 
        [-number_layer NUMBER_LAYER] [-hidden_size HIDDEN_SIZE] [-num_attention_head NUM_ATTENTION_HEAD]
        [-filter_size FILTER_SIZE] [-full_attention FULL_ATTENTION]
        model_output_dir abs_path_to_lasertagger abs_path_to_bert training_file tuning_file embedding_type

    positional arguments:
      model_output_dir      the directory of the model output
      abs_path_to_lasertagger
                            absolute path to the folder where the lasertagger scripts are located
      abs_path_to_bert      absolute path to the folder where the pretrained BERT is located
      training_file         path to training samples
      tuning_file           path to tuning samples
      embedding_type        type of embedding. Must be one of [Normal, POS, Sentence].
                            Normal: segment id is all zero. POS: part of speech tagging. Sentence: sentence tagging.

    optional arguments:
      -h, --help            show this help message and exit
      -vocab_size VOCAB_SIZE
                            vocab size. default = 500
      -train_batch_size TRAIN_BATCH_SIZE
                            batch size during training. default = 32
      -learning_rate LEARNING_RATE
                            The initial learning rate for Adam. default = 3e-5
      -num_train_epochs NUM_TRAIN_EPOCHS
                            Total number of training epochs to perform. default = 3
      -warmup_proportion WARMUP_PROPORTION
                            Proportion of training to perform linear learning rate warmup for. default = 0.1
      -max_input_examples MAX_INPUT_EXAMPLES
                            number of training examples to use in the vocab optimization. default is all training data.
      -train                If added, skip preprocessing and start training
      -export               If added, skip preprocessing and training, and start exporting to bucket
      -use_tpu              If added, will use tpu for training
      -gbucket GBUCKET      The gcp bucket where cloud TPU will store intermediary outputs to
      -t2t T2T              If True, use autoregressive version of LaserTagger. If false, use, feed-forward version of LaserTagger. 
                            Default is True.
      -number_layer NUMBER_LAYER
                            Number of hidden layers in the decoder. default = 1
      -hidden_size HIDDEN_SIZE
                            The size of the hidden layer size in the decoder. default=768
      -num_attention_head NUM_ATTENTION_HEAD
                            The number of attention heads in the decoder. default=4
      -filter_size FILTER_SIZE
                            The size of the filter in the decoder. default=3072
      -full_attention FULL_ATTENTION
                            Whether to use full attention in the decoder. default=false
      -masking              If added, numbers and symbols will be masked.
      -verb_deletion_loss VERB_DELETION_LOSS
                            The weight of verb deletion loss. Need to be >= 0. default=0. Cannot be set to a number 
                            other than 0 unless the embedding_type is POS.
      -add_tag_loss_weight ADD_TAG_LOSS_WEIGHT
                            The weight of loss for adding tags. default=1
      -delete_tag_loss_weight DELETE_TAG_LOSS_WEIGHT
                            The weight of loss for deleting tags. default=1
      -keep_tag_loss_weight KEEP_TAG_LOSS_WEIGHT
                            The weight of loss for keeping tags. default=1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model_output_dir", help="the directory of the model output")
    parser.add_argument("abs_path_to_lasertagger",
                        help="absolute path to the folder where the lasertagger scripts are located")
    parser.add_argument("abs_path_to_bert", help="absolute path to the folder where the pretrained BERT is located")
    parser.add_argument("training_file", help="path to training samples")
    parser.add_argument("tuning_file", help="path to tuning samples")
    parser.add_argument("embedding_type", help="type of embedding. Must be one of [Normal, POS, POS_concise, Sentence]. "
                        "Normal: segment id is all zero. POS: part of speech tagging. "
                        "POS_concise: POS tagging with a smaller set of tags. Sentence: sentence tagging.")
    
    parser.add_argument("-vocab_size", type=int, help="vocab size. default = 500", default=500)
    parser.add_argument("-train_batch_size", type=int, help="batch size during training. default = 32", default=32)
    parser.add_argument("-learning_rate", type=float, help="The initial learning rate for Adam. default = 3e-5", default=3e-5)
    parser.add_argument("-num_train_epochs", type=int, help="Total number of training epochs to perform. default = 3", default=3)
    parser.add_argument("-warmup_proportion", type=float,
    help="Proportion of training to perform linear learning rate warmup for. default = 0.1", default=0.1)
    parser.add_argument("-max_input_examples", type=int, help="number of training examples to use in the vocab optimization. "
    "default is all training data.")
    
    parser.add_argument("-train", action="store_true", help="If added, skip preprocessing and start training")
    parser.add_argument("-export", action="store_true",
    help="If added, skip preprocessing and training, and start exporting to bucket")
    parser.add_argument("-use_tpu", action="store_true", help="If added, will use tpu for training")
    parser.add_argument("-gbucket", help="The gcp bucket where cloud TPU will store intermediary outputs to")
    
    parser.add_argument("-t2t", type=bool, default=True, help="If True, use autoregressive version of LaserTagger. If false, "
                       "use, feed-forward version of LaserTagger. Default is True.")
    parser.add_argument("-number_layer", type=int, default=1, help="Number of hidden layers in the decoder. default = 1")
    parser.add_argument("-hidden_size", type=int, default=768, help="The size of the hidden layer size in the decoder. default=768")
    parser.add_argument("-num_attention_head", type=int, default=4, help="The number of attention heads in the decoder. default=4")
    parser.add_argument("-filter_size", type=int, default=3072, help="The size of the filter in the decoder. default=3072")
    parser.add_argument("-full_attention", type=bool, default=False, help="Whether to use full attention in the decoder. default=false")  
    
    parser.add_argument("-masking", action="store_true", help="If added, numbers and symbols will be masked.")
    parser.add_argument("-verb_deletion_loss", type=float, help="The weight of verb deletion loss. Need to be >= 0. default=0."
                        "Cannot be set to a number other than 0 unless the embedding_type is POS or POS_concise.", default=0)
    
    parser.add_argument("-add_tag_loss_weight", type=float, help="The weight of loss for adding tags. default=1", default=1)
    parser.add_argument("-delete_tag_loss_weight", type=float, help="The weight of loss for deleting tags. default=1", default=1)
    parser.add_argument("-keep_tag_loss_weight", type=float, help="The weight of loss for keeping tags. default=1", default=1)
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
