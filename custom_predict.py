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
"""Use trained LaserTagger model to make predictions."""

import argparse
import csv
import os
import subprocess

import language_tool_python
import nltk

from custom_post_processing_utils import post_processing
import preprocess_utils

TEMP_FOLDER_NAME = "temp_custom_predict"
TEMP_FOLDER_PATH = "~/" + TEMP_FOLDER_NAME
GCP_BUCKET = "gs://trained_models_yechen/"


def __download_models(list_of_models):
  """Download trained models from Google Cloud Bucket.

    Args:
        list_of_models: a list of trained models
    Raises:
        Exception: if the specified trained model does not exist at the GCP storage bucket 
    """
  for model in list_of_models:
    if os.path.isdir(model):
      print("-------model", model, "exists-------")
    else:
      print("-------downloading model", model, "-------")
      try:
        os.environ["model_name"] = GCP_BUCKET + model
        subprocess.call(['gsutil', '-m', 'cp', '-r', GCP_BUCKET + model, "./"],
                        cwd=os.path.expanduser('~'))
      except:
        raise Exception(
            "Model", model,
            "download failed. Check whether this model exists in the folder" +
            GCP_BUCKET)


def __validate_scripts(args):
  """Download LaserTagger and Bert scripts, and validate input file.

    Args:
        args: Command line arguments
    Raises:
        Exception: If intput file path does not exist
        Exception: If LaserTagger folder does not exist
        Exception: If "bert" folder does not exist within the LaserTagger folder
        Exception: If pretrained Bert model is not found
    """
  nltk.download('punkt')

  if not os.path.isfile(os.path.expanduser(args.path_to_input_file)):
    raise Exception("Input file not found.")

  if not os.path.isdir(os.path.expanduser(args.abs_path_to_lasertagger)):
    raise Exception("LaserTagger not found.")
  if not os.path.isdir(
      os.path.expanduser(args.abs_path_to_lasertagger + "/bert")):
    raise Exception("Bert not found inside the LaserTagger folder.")

  if not os.path.isdir(os.path.expanduser(args.abs_path_to_bert)):
    raise Exception("Pretrained Bert model not found.")


def __clean_up():
  """Clean up the temporary folder. """
  subprocess.call(['rm', '-rf', TEMP_FOLDER_NAME], cwd=os.path.expanduser('~'))


def __preprocess_input(input_file_path, whether_score):
  """Preprocess the input sentences to fit the format of lasertagger input.

    Args:
        input_file_path: the absolute path to the input file
        whether_score: whether scoring is needed. If scoring is needed, two columns are expected in the input file.
        
    Returns:
        sentences: a list of input sentences
        summaries: a list of summaries
        
    Raises:
        Exception: If scoring is required, but target is not found in the input file
    """
  if not os.path.isfile(os.path.expanduser(input_file_path)):
    __clean_up()
    raise Exception("The input file does not exist")
  print("-------Cleaning inputs-------")
  tsv_file = open(input_file_path)
  read_tsv = csv.reader(tsv_file, delimiter="\t")

  sentences = []
  summaries = []
  for row in read_tsv:
    sentences.append(row[0])
    if whether_score:
      try:
        summaries.append(row[1])
      except IndexError:
        tsv_file.close()
        __clean_up()
        raise Exception(
            "Whether_score is true. Expected target but only found one column in the input."
        )
  tsv_file.close()

  cleaned_sentences = preprocess_utils.text_strip(sentences)
  if whether_score:
    cleaned_summaries = preprocess_utils.text_strip(summaries)
  else:
    cleaned_summaries = cleaned_sentences

  cleaned_sentences, cleaned_summaries = preprocess_utils.delete_empty_entry(
      cleaned_sentences, cleaned_summaries)
  preprocess_utils.validate_dataset(cleaned_sentences, cleaned_summaries)

  spaced_sentences = preprocess_utils.tokenize_with_space(cleaned_sentences)
  if whether_score:
    spaced_summaries = preprocess_utils.tokenize_with_space(cleaned_summaries)
  else:
    spaced_summaries = spaced_sentences

  preprocess_utils.delete_empty_entry(spaced_sentences, spaced_summaries)

  return spaced_sentences, spaced_summaries


def main(args):
  """
    Compute predictions and scores for inputs using specified BERT model and LaserTagger model. 
    
    Read input sentences from input_file_path, convert the sentences to predicted summaries using pretrained
    models whose names are specified in the list_of_models, and compute exact score and SARI score if whether_score is
    true. The predictions are stored in an output file pred.tsv. If scores are computed, the scores are stored in
    an output file score.tsv.

    Args:
        args: command line arguments.
    """

  whether_score = args.score
  input_file_path = args.path_to_input_file
  list_of_models = args.models
  whether_grammar = args.grammar

  __download_models(list_of_models)
  __validate_scripts(args)

  __clean_up()
  subprocess.call(['mkdir', TEMP_FOLDER_NAME], cwd=os.path.expanduser('~'))
  spaced_sentences, spaced_summaries = __preprocess_input(
      input_file_path, whether_score)

  with open(os.path.expanduser(TEMP_FOLDER_PATH + "/cleaned_data.tsv"),
            'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i, sentence in enumerate(spaced_sentences):
      tsv_writer.writerow([sentence, spaced_summaries[i]])
  print("-------Number of input is", len(spaced_sentences), "-------")

  # calculate and print predictions to output file
  for model in list_of_models:
    print("------Running on model", model, "-------")
    prediction_command = [
        'python',
        os.path.expanduser(args.abs_path_to_lasertagger) + '/predict_main.py',
        "--input_format=wikisplit",
        "--label_map_file=./" + model + "/label_map.txt",
        "--input_file=" + "./" + TEMP_FOLDER_NAME + "/cleaned_data.tsv",
        "--saved_model=./" + model + "/export_model", "--vocab_file=" +
        os.path.expanduser(args.abs_path_to_bert) + "/vocab.txt",
        "--output_file=" + "./" + TEMP_FOLDER_NAME + "/output_" + model +
        ".tsv", "--embedding_type=" + args.embedding_type,
        "--batch_size=" + str(args.batch_size)
    ]
    if args.masking:
      prediction_command.append("--enable_masking=true")
    subprocess.call(prediction_command, cwd=os.path.expanduser("~"))
    print("------Completed running on model", model, "-------")

  output_row_list = []

  model = list_of_models[0]
  output_row = ["original"]
  tsv_file = open(
      os.path.expanduser(TEMP_FOLDER_PATH + "/output_" + model + ".tsv"))
  read_tsv = csv.reader(tsv_file, delimiter="\t")
  for row in read_tsv:
    output_row.append(row[0])
  output_row_list.append(output_row)

  for model in list_of_models:
    output_row = [model]
    tsv_file = open(
        os.path.expanduser(TEMP_FOLDER_PATH + "/output_" + model + ".tsv"))
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv:
      output_row.append(post_processing(row[1]))
    output_row_list.append(output_row)

  if whether_grammar:
    tool = language_tool_python.LanguageTool('en-US')
    for model in list_of_models:
      output_row = [model + "_corrected"]
      tsv_file = open(
          os.path.expanduser(TEMP_FOLDER_PATH + "/output_" + model + ".tsv"))
      read_tsv = csv.reader(tsv_file, delimiter="\t")
      for row in read_tsv:
        output_row.append(tool.correct(post_processing(row[1])))
      output_row_list.append(output_row)

  if whether_score:
    model = list_of_models[0]
    output_row = ["target"]
    tsv_file = open(
        os.path.expanduser(TEMP_FOLDER_PATH + "/output_" + model + ".tsv"))
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv:
      output_row.append(row[2])
    output_row_list.append(output_row)

  with open(os.path.expanduser("~/pred.tsv"), 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for i in range(len(output_row)):
      this_row = []
      for row in output_row_list:
        this_row.append(row[i])
      tsv_writer.writerow(this_row)
  print("------Predictions written out to pred.tsv------")

  # calculate and print scores to output file if whether_score is True
  if whether_score:
    for model in list_of_models:
      print("------Calculating score for model", model, "-------")
      f = open(
          os.path.expanduser(TEMP_FOLDER_PATH + "/score_" + model + ".txt"),
          "w")
      subprocess.call([
          'python',
          os.path.expanduser(args.abs_path_to_lasertagger) + '/score_main.py',
          "--prediction_file=" + "./" + TEMP_FOLDER_NAME + "/output_" + model +
          ".tsv"
      ],
                      cwd=os.path.expanduser('~'),
                      stdout=f)

    output_row_list = []
    output_row = [
        "score", "Exact score", "SARI score", "KEEP score", "ADDITION score",
        "DELETION score"
    ]
    output_row_list.append(output_row)

    for model in list_of_models:
      output_row = [model]
      f = open(
          os.path.expanduser(TEMP_FOLDER_PATH + "/score_" + model + ".txt"))
      lines = f.readlines()
      for line in lines:
        output_row.append(line.split()[2])
      output_row_list.append(output_row)

    with open(os.path.expanduser("~/score.tsv"), 'wt') as out_file:
      tsv_writer = csv.writer(out_file, delimiter='\t')
      for i in range(len(output_row)):
        this_row = []
        for row in output_row_list:
          this_row.append(row[i])
        tsv_writer.writerow(this_row)
      print("------Scores written out to score.tsv------")

  __clean_up()


if __name__ == "__main__":
  """Compute predictions and scores for inputs using specified BERT model and LaserTagger model. 
    
    usage: custom_predict.py [-h] [-score] path_to_input_file abs_path_to_lasertagger 
                                           abs_path_to_bert models [models ...]
    positional arguments:
      path_to_input_file    the directory of the model output
      abs_path_to_lasertagger
                            absolute path to the folder where the lasertagger scripts are located
      abs_path_to_bert      absolute path to the folder where the pretrained BERT is located
      models                the name of trained models
      embedding_type        type of embedding. Must be one of [Normal, POS, Sentence]. 
                            Normal: segment id is all zero. POS: part of speech tagging. 
                            Sentence: sentence tagging.
                            
    optional arguments:
      -h, --help            show help message and exit
      -score                If added, compute scores for the predictions
      -grammar              If added, automatically apply grammar check on predictions
      -masking              If added, numbers and symbols will be masked.
      -batch_size           The batch size of prediction. Default=1.
    """
  parser = argparse.ArgumentParser()
  parser.add_argument("path_to_input_file",
                      help="the directory of the model output")
  parser.add_argument(
      "abs_path_to_lasertagger",
      help=
      "absolute path to the folder where the lasertagger scripts are located")
  parser.add_argument(
      "abs_path_to_bert",
      help="absolute path to the folder where the pretrained BERT is located")
  parser.add_argument('models', help="the name of trained models", nargs='+')
  parser.add_argument(
      "embedding_type",
      help="type of embedding. Must be one of [Normal, POS, Sentence]. "
      "Normal: segment id is all zero. POS: part of speech tagging. Sentence: sentence tagging."
  )

  parser.add_argument("-score",
                      action="store_true",
                      help="If added, compute scores for the predictions")
  parser.add_argument(
      "-grammar",
      action="store_true",
      help="If added, automatically apply grammar check on predictions")
  parser.add_argument("-batch_size",
                      default=1,
                      type=int,
                      help="The batch size of prediction. Default=1.")
  parser.add_argument("-masking",
                      action="store_true",
                      help="If added, numbers and symbols will be masked.")

  arguments = parser.parse_args()
  if arguments.embedding_type not in [
      "Normal", "POS", "Sentence", "POS_concise"
  ]:
    raise ValueError(
        "Embedding_type must be Normal, POS, POS_concise, or Sentence")

  main(arguments)
