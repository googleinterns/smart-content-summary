import csv
import os
import subprocess
import sys

import pandas as pd

import preprocess_utils
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


PREPROCESSED_FILE_PATH = "~/preprocessed_reddit_dataset.tsv"
TRAIN_FILE_PATH = "~/train_reddit_dataset.tsv"
TUNE_FILE_PATH = "~/tune_reddit_dataset.tsv"
VALID_FILE_PATH = "~/valid_reddit_dataset.tsv"


def main(argv):
    if len(argv) != 2:
        raise Exception("Usage: python preprocess_reddit_dataset num_of_tuning_samples num_of_validation_samples")
    
    try:
        num_of_tuning_sam = int(argv[0])
        num_of_valid_sam = int(argv[1])
    except ValueError:
        raise Exception("Number of samples must be non-negative integers")
    
    if not os.path.isfile(os.path.expanduser(PREPROCESSED_FILE_PATH)):
        ds = tfds.load('reddit_tifu', split='train', shuffle_files=True)
        
        sentences = []
        summaries = []
        for row in ds:
            summary = row["title"]
            sentence = row["tldr"]

            sentences.append(sentence.numpy().decode('UTF-8'))
            summaries.append(summary.numpy().decode('UTF-8'))
    
        cleaned_sentences = preprocess_utils.text_strip(sentences)
        cleaned_summaries = preprocess_utils.text_strip(summaries)

        cleaned_sentences, cleaned_summaries = preprocess_utils.delete_empty_entry(cleaned_sentences, cleaned_summaries)
        
        preprocess_utils.validate_dataset(cleaned_sentences, cleaned_summaries)
        print("Number of samples is", len(cleaned_sentences))

        preprocess_utils.calculate_stats(cleaned_sentences, cleaned_summaries)
        spaced_sentences = preprocess_utils.tokenize_with_space(cleaned_sentences)
        spaced_summaries = preprocess_utils.tokenize_with_space(cleaned_summaries)

        with open(os.path.expanduser(PREPROCESSED_FILE_PATH), 'wt') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                for i in range(len(spaced_sentences)):
                    tsv_writer.writerow([spaced_sentences[i], spaced_summaries[i]])
        print("-------Preprocessed data saved to", PREPROCESSED_FILE_PATH, "-------")
    else:
        print("-------Preprocessed data exists. Now splitting dataset.-------")
    print("-------Now splitting dataset.-------")
    preprocess_utils.split_dataset(TRAIN_FILE_PATH, TUNE_FILE_PATH, VALID_FILE_PATH, PREPROCESSED_FILE_PATH,
                                   num_of_tuning_sam, num_of_valid_sam, whether_shuffle=False)

    

if __name__ == "__main__":
    main(sys.argv[1:])
