import csv
import os
import sys

import preprocess_utils

GRADING_COMMENTS = ["Most important meaning Flawless language", "Most important meaning Minor errors", \
                    "Most important meaning Disfluent or incomprehensible", "Much meaning Flawless language", \
                    "Much meaning Minor errors", "Much meaning Disfluent or incomprehensible", \
                    "Little or none meaning Flawless language", "Little or none meaning Minor errors", \
                    "Little or none meaning Disfluent or incomprehensible"]
GRADING_NUMBER = ["6", "7", "9", "11", "12", "14", "21", "22", "24"]
EXCLUSION_NUMBER = ["9", "14", "21", "22", "24"]

PREPROCESSED_FILE_PATH = "~/preprocessed_MS_dataset.tsv"
TRAIN_FILE_PATH = "~/train_MS_dataset.tsv"
TUNE_FILE_PATH = "~/tune_MS_dataset.tsv"
VALID_FILE_PATH = "~/valid_MS_dataset.tsv"


def __process_row(row):
    """Split a row into the original sentence, its corresponding summary or summaries and corresponding ratings of the
    summary or summaries.
    Args:
        row: a row in the MS dataset
    Returns:
        current_original_sentence: the original sentence of the row
        current_shortened_sentences_list: a list of summaries corresponding to the current_original_sentence
        current_shortened_ratings_list: the ratings of the summaries in current_shortened_sentences_list
        count_excluded: number of summaries excluded on the row due to low ratings
    """
    count_excluded = 0
    row_flattened = []
    for i in range(len(row)):
        splitted_row = row[i].split(" ||| ")
        for j in range(len(splitted_row)):
            if (splitted_row[j] not in GRADING_COMMENTS):
                row_flattened.append(splitted_row[j])

    current_original_sentence = row_flattened[2]
    current_shortened_sentences_list = []
    current_shortened_ratings_list = []

    this_shortened_sentence = row_flattened[3]
    this_shortened_sentence_rating = []
    for i in range(4, len(row_flattened)):
        if i + 1 == len(row_flattened):
            this_shortened_sentence_rating.append(row_flattened[i])
            this_shortened_sentence_rating = this_shortened_sentence_rating[2:]
            if len(this_shortened_sentence_rating) == 0 or \
                    len(set(this_shortened_sentence_rating).intersection(set(EXCLUSION_NUMBER))) / len(
                this_shortened_sentence_rating) < 0.5:
                current_shortened_sentences_list.append(this_shortened_sentence)
                current_shortened_ratings_list.append(this_shortened_sentence_rating)
            else:
                count_excluded += 1

        elif not row_flattened[i].isnumeric() and not row_flattened[i].split(";")[0].isnumeric():
            this_shortened_sentence_rating = this_shortened_sentence_rating[2:]
            if len(this_shortened_sentence_rating) == 0 or \
                    len(set(this_shortened_sentence_rating).intersection(set(EXCLUSION_NUMBER))) / len(
                this_shortened_sentence_rating) < 0.5:
                current_shortened_sentences_list.append(this_shortened_sentence)
                current_shortened_ratings_list.append(this_shortened_sentence_rating)
                this_shortened_sentence = row_flattened[i]
                this_shortened_sentence_rating = []
            else:
                count_excluded += 1
        else:
            this_shortened_sentence_rating.append(row_flattened[i])
    assert (len(current_shortened_sentences_list) == len(current_shortened_ratings_list))
    return current_original_sentence, current_shortened_sentences_list, current_shortened_ratings_list, count_excluded


def __process_file(file_path):
    """Process a tsv file in the MS dataset.
    Args:
        file_path: direct path to the tsv file
    Returns:
        sentences: a list of original sentences
        summaries: a list of summaries corresponding to the original sentences
        ratings: a list of ratings of the summaries
        count_excluded: the number of sentence-summary pairs excluded in the file due to low rating
    """
    tsv_file = open(os.path.expanduser(file_path))
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    sentences = []
    summaries = []
    ratings = []
    count_excluded = 0
    for row in read_tsv:
        row_sentence, row_summary, row_rating, row_count_excluded = __process_row(row)

        for i in range(len(row_summary)):
            sentences.append(row_sentence)
            summaries.append(row_summary[i])
            ratings.append(row_rating[i])
        count_excluded += row_count_excluded

    return sentences, summaries, ratings, count_excluded


def main(argv):
    """Preprocess the Microsoft dataset."""
    if len(argv) != 3:
        raise Exception(
            "Usage: python preprocess_MS_dataset absolute_path_to_RawData_dir num_of_tuning_samples num_of_validation_samples")

    data_dir = argv[0]

    try:
        num_of_tuning_sam = int(argv[1])
        num_of_valid_sam = int(argv[2])
    except ValueError:
        raise Exception("Number of samples must be non-negative integers")

    if not os.path.isfile(os.path.expanduser(PREPROCESSED_FILE_PATH)):
        train_data_dir = data_dir + "/train.tsv"
        train_sentences, train_summaries, train_ratings, train_excluded = __process_file(train_data_dir)
        test_data_dir = data_dir + "/test.tsv"
        test_sentences, test_summaries, test_ratings, test_excluded = __process_file(test_data_dir)
        valid_data_dir = data_dir + "/valid.tsv"
        valid_sentences, valid_summaries, valid_ratings, valid_excluded = __process_file(valid_data_dir)

        tot_sentences = train_sentences + test_sentences + valid_sentences
        tot_summaries = train_summaries + test_summaries + valid_summaries
        tot_ratings = train_ratings + test_ratings + valid_ratings
        tot_excluded = train_excluded + test_excluded + valid_excluded

        cleaned_sentences = preprocess_utils.text_strip(tot_sentences)
        cleaned_summaries = preprocess_utils.text_strip(tot_summaries)

        cleaned_sentences, cleaned_summaries = preprocess_utils.delete_empty_entry(cleaned_sentences, cleaned_summaries)
        preprocess_utils.validate_dataset(cleaned_sentences, cleaned_summaries)
        print("Number of samples is", len(cleaned_sentences))
        print("Total number of excluded sample is", tot_excluded)

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
