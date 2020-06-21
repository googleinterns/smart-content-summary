import csv
import os
import random
import re

import nltk
import numpy as np

nltk.download('punkt')
"""Utilities for preprocessing input texts for LaserTagger."""


def text_strip(column):
    """Preprocess input texts.

    Args:
      column: one column of input texts.
      
    Returns:
      the cleaned version of the text column.
    """
    cleaned_col = []
    for row in column:
        # remove escape characters
        row = re.sub("(\\t)", ' ', str(row))
        row = re.sub("(\\r)", ' ', str(row))
        row = re.sub("(\\n)", ' ', str(row))

        # remove redundant special characters
        row = re.sub("(__+)", ' ', str(row))
        row = re.sub("(--+)", ' ', str(row))
        row = re.sub("(~~+)", ' ', str(row))
        row = re.sub("(\+\++)", ' ', str(row))
        row = re.sub("(\.\.+)", ' ', str(row))

        # remove - at end of words(not between)
        row = re.sub("(\-\s+)", ' ', str(row))
        # remove : at end of words(not between)
        row = re.sub("(\:\s+)", ' ', str(row))
        # remove any single charecters hanging between 2 spaces
        row = re.sub("(\s+.\s+)", ' ', str(row))

        # remove multiple spaces
        row = re.sub("(\s+)", ' ', str(row))

        # remove any single charecters hanging between 2 spaces
        row = re.sub("(\s+.\s+)", ' ', str(row))

        cleaned_col.append(row)

    return cleaned_col


def validate_dataset(sentences, summaries):
    """Validate that the dataset has same number of sentences and summaries, and that it does not contain empty entry.

        Args:
          sentences: one column of input texts.
          summaries: one column of summaries corresponding to sentences.
    """
    if len(sentences) != len(summaries):
        raise Exception("The number of original sentences does not match the number of summaries.")

    for sentence in sentences:
        if sentence.split() == 0:
            raise Exception("Original sentences contains empty examples.")

    for sentence in summaries:
        if sentence.split() == 0:
            raise Exception("Summaries contains empty examples.")


def calculate_stats(sentences, summaries):
    """For the input sentences and their corresponding summaries, calculate relevant statistics: average, maximum, and
    minimum number of words in original sentences and their summaries; average number of sentences in each input
    sentences and summaries; average number of words that are in the summary but not in the corresponding original
    sentence; and average compression ratio.

    Args:
      sentences: a column of input sentences.
      summaries: a column of summaries corresponding to the sentences in the
                  sentences column. 
    """

    print("-------Calculating statistics-------")
    validate_dataset(sentences, summaries)

    # average, maximum, and minimum number of words in original sentences and their summaries
    count_words = []
    for sentence in sentences:
        count_words.append(len(sentence.split()))
    count_words = np.array(count_words)
    print("Average word count of original sentence is", "{:.2f}".format(np.mean(count_words)), "( std:",
          "{:.2f}".format(np.std(count_words)), ")")
    print("Max word count is", np.max(count_words))
    print("Min word count is", np.min(count_words))

    count_words = []
    for summary in summaries:
        count_words.append(len(summary.split()))
    count_words = np.array(count_words)
    print("Average word count of shortened sentence is", "{:.2f}".format(np.mean(count_words)), "( std:",
          "{:.2f}".format(np.std(count_words)), ")")
    print("Max Length is", np.max(count_words))
    print("Min Length is", np.min(count_words))

    # average number of sentences in each input sentences and summaries; average number of words that are in the summary
    # but not in the corresponding original sentence
    count_diff = []
    count_sentences = []
    for i in range(len(sentences)):
        tokens_sentence = nltk.word_tokenize(sentences[i])
        tokens_sentence = [x.lower() for x in tokens_sentence]
        tokens_headline = nltk.word_tokenize(summaries[i])
        tokens_headline = [x.lower() for x in tokens_headline]
        count_diff.append(len(list(set(tokens_headline) - set(tokens_sentence))))
        count_sentences.append(
            np.max([tokens_sentence.count(".") + tokens_sentence.count("!") + tokens_sentence.count("?"), 1]))
    count_sentences = np.array(count_sentences)
    count_diff = np.array(count_diff)
    print("On average, there are", "{:.2f}".format(np.mean(count_sentences)), "sentences in each original text",
          "( std:", "{:.2f}".format(np.std(count_sentences)), ")")
    print("On average, there are", "{:.2f}".format(np.mean(count_diff)),
          "words in each shortened sentence that are not in the original sentence.",
          "( std:", "{:.2f}".format(np.std(count_diff)), ")")

    # average compression ratio
    compression_ratio = []
    for i in range(len(sentences)):
        compression_ratio.append(len(summaries[i].split()) / len(sentences[i].split()))
    compression_ratio = np.array(compression_ratio)
    print("The average compression ratio is", "{:.2f}".format(np.mean(compression_ratio)), "( std:",
          "{:.2f}".format(np.std(compression_ratio)), ")")


def tokenize_with_space(sentences):
    """Tokenize the input text with spaces separating the tokens.

    Args:
      sentences: a column of input sentences.
    
    Returns:
      a column of tokenized texts with spaces separating the tokens.
    """
    spaced_sentences = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        spaced_sentence = ""
        for j in range(len(tokens) - 1):
            spaced_sentence += tokens[j] + " "
        spaced_sentence += tokens[-1]
        spaced_sentences.append(spaced_sentence)

    return spaced_sentences


def split_dataset(train_path, tune_path, valid_path, preprocessed_path, num_of_tuning_sam, num_of_valid_sam,
                  whether_shuffle):
    """Split the dataset into training, tuning, and validation sets, and store each in a file.

        Args:
          train_path: path to store the training set.
          tune_path: path to store the tuning set.
          valid_path: path to store the validation set.
          preprocessed_path: path where the preprocessed dataset is store.
          num_of_tuning_sam: number of tuning samples.
          num_of_valid_sam: number of validation samples.
          whether_shuffle: whether the dataset needs to be shuffled before splitting.

        Side effects:
          Training set is stored in the path specified by train_path.
          Tuning set is stored in the path specified by tune_path.
          Validation set is stored in the path specified by valid_path.
    """
    sentences = []
    summaries = []
    with open(os.path.expanduser(preprocessed_path), 'r') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line in tsv_reader:
            sentences.append(line[0])
            summaries.append(line[1])

    validate_dataset(sentences, summaries)

    if whether_shuffle:
        print("-------Shuffling the dataset-------")
        sentence_shuffled = []
        summary_shuffled = []
        index_shuffled = list(range(len(sentences)))
        random.shuffle(index_shuffled)
        for i in index_shuffled:
            sentence_shuffled.append(sentences[i])
            summary_shuffled.append(summaries[i])
        sentences = sentence_shuffled
        summaries = summary_shuffled

    if num_of_tuning_sam + num_of_valid_sam > len(sentences):
        raise Exception("The number of tuning and validation samples together exceeds the total sample size of " + str(
            len(sentences)))

    with open(os.path.expanduser(tune_path), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for i in range(num_of_tuning_sam):
            tsv_writer.writerow([sentences[i], summaries[i]])
    print("-------", num_of_tuning_sam, "tuning samples wrote to", tune_path, "-------")

    with open(os.path.expanduser(valid_path), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for i in range(num_of_tuning_sam, num_of_tuning_sam + num_of_valid_sam):
            tsv_writer.writerow([sentences[i], summaries[i]])
    print("-------", num_of_valid_sam, "validation samples wrote to", valid_path, "-------")

    with open(os.path.expanduser(train_path), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for i in range(num_of_tuning_sam + num_of_valid_sam, len(sentences)):
            tsv_writer.writerow([sentences[i], summaries[i]])
    print("-------", len(sentences) - num_of_tuning_sam - num_of_valid_sam, "tuning samples wrote to", train_path,
          "-------")
    
    
def delete_empty_entry(sentences, summaries):
    """ Delete empty entries from the dataset.
        Args:
         sentences: a column of input sentences.
         summaries: a column of summaries corresponding to the sentences in the
                   sentences column. 
        Returns:
         sentences: a column of input sentences with empty entries deleted 
         summaries: a column of summaries with empty entries deleted
    """
    empty_index = []
    for i, sentence in enumerate(sentences):
        if len(sentence.split()) == 0:
            empty_index.append(i)
    for i, sentence in enumerate(summaries):
        if len(sentence.split()) == 0:
            empty_index.append(i)
    empty_index = sorted(list(set(empty_index)), reverse=True)
    for index in empty_index:
        del sentences[index]
        del summaries[index]
    
    print("-------Deleted", len(empty_index), "empty entries-------")
    return sentences, summaries
