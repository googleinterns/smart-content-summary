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
import csv
import os
import subprocess
import sys

import preprocess_utils

TMP_FOLDER_NAME = "tmp_preprocess_google_dataset"
TMP_FOLDER_DIR = "~/" + TMP_FOLDER_NAME
DATASET_NAME = "sentence-compression"
DATASET_DIR = "~/" + DATASET_NAME + "/data"
PREPROCESSED_FILE_PATH = "~/preprocessed_google_dataset.tsv"
TRAIN_FILE_PATH = "~/train_google_dataset.tsv"
TUNE_FILE_PATH = "~/tune_google_dataset.tsv"
VALID_FILE_PATH = "~/valid_google_dataset.tsv"


def __clean_up():
    """ Clean up temporary intermediate files."""
    subprocess.call(['rm', '-rf', TMP_FOLDER_NAME], cwd=os.path.expanduser('~'))


def __download_data():
    """ Download Google dataset from Github."""
    if not os.path.isdir(os.path.expanduser("~/" + DATASET_NAME)):
        print("-------Downloading dataset-------")
        subprocess.call("git clone https://github.com/google-research-datasets/sentence-compression.git ".split(),
                        cwd=os.path.expanduser('~'))
        __unzip_data(DATASET_DIR)
    else:
        print("-------Updating dataset-------")
        output = subprocess.check_output("git pull".split(), cwd=os.path.expanduser("~/" + DATASET_NAME))
        if output != b"Already up-to-date.\n":
            __unzip_data(DATASET_DIR)
    print("-------Dataset up to date-------")


def __unzip_data(dataset_dir):
    """Unzip files in the Google dataset.
    Args:
        dataset_dir: the directory where the Google dataset is in.
    """
    print("-------Unzipping dataset-------")
    for i in range(1, 11):
        subprocess.call(["gunzip", "sent-comp.train" + str(i).zfill(2) + ".json.gz"],
                        cwd=os.path.expanduser(dataset_dir))
    subprocess.call("gunzip comp-data.eval.json.gz".split(), cwd=os.path.expanduser(dataset_dir))


def __format_data():
    """ Format the dataset and clean up special characters.
    Returns:
        cleaned_sentences: a column of cleaned input sentences
        cleaned_summaries: a column of cleaned summaries corresponding to the input sentences
    """
    print("-------Processing original sentences-------")
    for i in range(1, 11):
        subprocess.call('cat sent-comp.train' +
                        str(i).zfill(2) + '.json | grep \'"sentence":\' > ~/' +
                        TMP_FOLDER_NAME + '/train' + str(i) + '.txt', shell=True,
                        cwd=os.path.expanduser(DATASET_DIR))

    subprocess.call('cat comp-data.eval.json | grep \'"sentence":\' > ~/' + TMP_FOLDER_NAME + '/train11.txt',
                    shell=True, cwd=os.path.expanduser(DATASET_DIR))

    sentences = []
    for i in range(1, 12):
        file_name = os.path.expanduser(TMP_FOLDER_NAME) + '/train' + str(i) + '.txt'
        f = open(file_name, "r")
        odd_line = True
        for line in f:
            if odd_line:
                sentences.append(line[17:-3])
            odd_line = not odd_line
        f.close()
    cleaned_sentences = preprocess_utils.text_strip(sentences)

    print("-------Processing summaries-------")
    for i in range(1, 11):
        subprocess.call('cat sent-comp.train' + str(i).zfill(
            2) + '.json | grep \'"headline":\' > ~/' + TMP_FOLDER_NAME + '/train' + str(i) + '.txt', shell=True,
                        cwd=os.path.expanduser(DATASET_DIR))

    subprocess.call('cat comp-data.eval.json | grep \'"headline":\' > ~/' + TMP_FOLDER_NAME + '/train11.txt',
                    shell=True, cwd=os.path.expanduser(DATASET_DIR))

    summaries = []
    for i in range(1, 12):
        file_name = os.path.expanduser(TMP_FOLDER_NAME) + '/train' + str(i) + '.txt'
        f = open(file_name, "r")
        for line in f:
            summaries.append(line[15:-3])
        f.close()

    cleaned_summaries = preprocess_utils.text_strip(summaries)
    cleaned_sentences, cleaned_summaries = preprocess_utils.delete_empty_entry(cleaned_sentences, cleaned_summaries)
    preprocess_utils.validate_dataset(cleaned_sentences, cleaned_summaries)
    print("Number of samples is", len(cleaned_sentences))

    return cleaned_sentences, cleaned_summaries


def main(argv):
    """Preprocess the Google dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("num_of_tuning", help="Number of tuning samples", type=int)
    parser.add_argument("num_of_validation", help="Number of validation samples", type=int)

    args = parser.parse_args()
    num_of_tuning_sam = args.num_of_tuning
    num_of_valid_sam = args.num_of_validation

    if num_of_valid_sam < 0 or num_of_tuning_sam < 0:
        raise Exception("Number of samples must be non-negative integers")

    if not os.path.isfile(os.path.expanduser(PREPROCESSED_FILE_PATH)):
        __clean_up()
        subprocess.call(['mkdir', TMP_FOLDER_NAME], cwd=os.path.expanduser('~'))
        __download_data()
        cleaned_sentences, cleaned_summaries = __format_data()
        preprocess_utils.calculate_stats(cleaned_sentences, cleaned_summaries)
        spaced_sentences = preprocess_utils.tokenize_with_space(cleaned_sentences)
        spaced_summaries = preprocess_utils.tokenize_with_space(cleaned_summaries)
        __clean_up()

        with open(os.path.expanduser(PREPROCESSED_FILE_PATH), 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for i in range(len(spaced_sentences)):
                tsv_writer.writerow([spaced_sentences[i], spaced_summaries[i]])
        print("-------Preprocessed data saved to", PREPROCESSED_FILE_PATH, "-------")
        print("-------Now splitting dataset.-------")
    else:
        print("-------Preprocessed data exists. Now splitting dataset.-------")

    preprocess_utils.split_dataset(TRAIN_FILE_PATH, TUNE_FILE_PATH, VALID_FILE_PATH, PREPROCESSED_FILE_PATH,
                                   num_of_tuning_sam, num_of_valid_sam, whether_shuffle_entire_set=True,
                                   whether_shuffle_individual_file=True)


if __name__ == "__main__":
    main(sys.argv[1:])
