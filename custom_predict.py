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

import sys
import os
import subprocess
import csv
import re
import nltk

import preprocess_utils

TMP_FOLDER_NAME = "tmp_custom_predict"
TMP_FOLDER_PATH = "~/" + TMP_FOLDER_NAME

def __download_models(list_of_models):
    for model in list_of_models:
        if os.path.isdir(model):
            print("-------model", model, "exists-------")
        else:
            print("-------downloading model", model,"-------")
            try:
                os.environ["model_name"] = "gs://trained_models_yechen/" + model
                subprocess.call(['gsutil', '-m', 'cp', '-r', "gs://trained_models_yechen/" + model, "./"], cwd=os.path.expanduser('~'))
            except:
                raise Exception("Model", model, "download failed. Check whether this model exists in the folder gs://trained_models_yechen",\
                               "Currently, trained models include: GG_500_AR, GG_800_AR, GG_1100_AR, MS_500_AR, MS_800_AR, MS_1100_AR.")

def __download_scripts():
    nltk.download('punkt')
    # download bert
    if not os.path.isdir("bert_vocab.txt"):
        print("-------downloading bert vocab-------")
        subprocess.check_output(['gsutil', 'cp', "gs://trained_models_yechen/bert_vocab.txt", "./"], cwd=os.path.expanduser('~'))
        print("-------completed downloading bert vocab-------")
    else:
        print("-------bert vocab exists-------")
    # git clone scripts
    if not os.path.isdir("lasertagger"):
        print("-------downloading lasertagger-------")
        subprocess.check_output(['git', 'clone', 'https://github.com/google-research/lasertagger.git'], cwd=os.path.expanduser('~'))
        subprocess.check_output(['git', 'clone', 'https://github.com/google-research/bert.git', './lasertagger/bert'], cwd=os.path.expanduser('~'))
        print("-------completed downloading bert-------")
    else:
        print("-------scripts exist-------")

def __clean_up():
    subprocess.call(['rm', '-rf', TMP_FOLDER_NAME], cwd=os.path.expanduser('~'))

def __preprocess_input(input_file_path, whether_score):
    if not os.path.isfile(os.path.expanduser(input_file_path)):
        clean_up()
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
                raise Exception("Whether_score is true. Expected target but only found one column in the input.")
    
    cleaned_sentences = preprocess_utils.text_strip(sentences)
    if whether_score:
        cleaned_summaries = preprocess_utils.text_strip(summaries)
    else:
        cleaned_summaries = cleaned_sentences
    
    cleaned_sentences, cleaned_summaries = preprocess_utils.delete_empty_entry(cleaned_sentences, cleaned_summaries)
    preprocess_utils.validate_dataset(cleaned_sentences, cleaned_summaries)
    
    spaced_sentences = preprocess_utils.tokenize_with_space(cleaned_sentences)
    if whether_score:
        spaced_summaries = preprocess_utils.tokenize_with_space(cleaned_summaries)
    else:
        spaced_summaries = spaced_sentences
        
    preprocess_utils.delete_empty_entry(spaced_sentences, spaced_summaries)

    
    with open(os.path.expanduser(TMP_FOLDER_PATH + "/cleaned_data.tsv"), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for i, sentence in enumerate(spaced_sentences):
            tsv_writer.writerow([sentence,spaced_summaries[i]])
    print("-------Number of input is", len(spaced_sentences), "-------")
    
    
def main(argv):
    if len(argv) < 3:
            raise Exception("Usage: python custom_predict.py whether_score(true or false) path/to/the_input_file.tsv 'Name-of-Model-1' 'Name-of-Model-2' ...")
    
    whether_score = argv[0]
    if whether_score == "true":
        whether_score = True
    elif whether_score == "false":
        whether_score = False
    else:
        raise Exception("whether_score should be true or false. Usage: python custom_predict.py whether_score(true or false) path/to/the_input_file.tsv 'Name-of-Model-1' 'Name-of-Model-2' ...")
    
    input_file_path = argv[1]
    list_of_models = argv[2:]
    if len(list_of_models) == 0:
        raise Exception("Need to provide at least one model. Usage: python custom_predict.py whether_score(true or false) path/to/the_input_file.tsv 'Name-of-Model-1' 'Name-of-Model-2' ...")

    __download_models(list_of_models)
    __download_scripts()
    
    __clean_up()
    subprocess.call(['mkdir', TMP_FOLDER_NAME], cwd=os.path.expanduser('~'))
    __preprocess_input(input_file_path, whether_score)
    
    # calculate and print predictions to output file 
    for model in list_of_models:
        print("------Running on model", model, "-------")
        subprocess.call(['python', 'lasertagger/predict_main.py', \
                                 "--input_format=wikisplit", \
                                 "--label_map_file=./"+model+"/label_map.txt",\
                                 "--input_file=" + "./" + TMP_FOLDER_NAME + "/cleaned_data.tsv", \
                                 "--saved_model=./" + model + "/export_model", \
                                 "--vocab_file=bert_vocab.txt",\
                                 "--output_file=" + "./" + TMP_FOLDER_NAME + "/output_" + model + ".tsv"], cwd=os.path.expanduser('~'))
        print("------Completed running on model", model, "-------")
     
    output_row_list = []
    
    model = list_of_models[0]
    output_row = ["original"]
    tsv_file = open(os.path.expanduser(TMP_FOLDER_PATH + "/output_" + model + ".tsv"))
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv:
        output_row.append(row[0])
    output_row_list.append(output_row)
    
    for model in list_of_models:
        output_row = [model]
        tsv_file = open(os.path.expanduser(TMP_FOLDER_PATH + "/output_" + model + ".tsv"))
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for row in read_tsv:
            output_row.append(row[1])
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
            f = open(os.path.expanduser(TMP_FOLDER_PATH + "/score_" + model + ".txt"), "w")
            subprocess.call(['python', 'lasertagger/score_main.py', \
                                     "--prediction_file=" + "./" + TMP_FOLDER_NAME + "/output_" + model + ".tsv"], \
                                     cwd=os.path.expanduser('~'), stdout=f)

        output_row_list = []
        output_row = ["score", "Exact score", "SARI score", "KEEP score", "ADDITION score", "DELETION score"]
        output_row_list.append(output_row)
        
        for model in list_of_models:
            output_row = [model]
            f = open(os.path.expanduser(TMP_FOLDER_PATH + "/score_" + model + ".txt"))
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
#     clean_up()

if __name__ == "__main__":
    main(sys.argv[1:])
