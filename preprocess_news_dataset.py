import csv
import os
import subprocess
import sys

import pandas as pd

import preprocess_utils


PREPROCESSED_FILE_PATH = "~/preprocessed_news_dataset.tsv"
TRAIN_FILE_PATH = "~/train_news_dataset.tsv"
TUNE_FILE_PATH = "~/tune_news_dataset.tsv"
VALID_FILE_PATH = "~/valid_news_dataset.tsv"


def main(argv):
    if len(argv) != 4:
        raise Exception("Usage: python preprocess_news_dataset absolute_path_to_dir_containing_news_summary.csv absolute_path_to_dir_containing_news_summary_more.csv num_of_tuning_samples num_of_validation_samples")
    
    data_dir_1 = argv[0] + "/news_summary.csv"
    data_dir_2 = argv[1] + "/news_summary_more.csv"
    
    try:
        num_of_tuning_sam = int(argv[2])
        num_of_valid_sam = int(argv[3])
    except ValueError:
        raise Exception("Number of samples must be non-negative integers")
    
        
    if not os.path.isfile(os.path.expanduser(PREPROCESSED_FILE_PATH)):
        if not os.path.isfile(os.path.expanduser(data_dir_1)):
            raise Exception ("Cannot find" + os.path.expanduser(data_dir_1) + ". If necessary, please download from https://www.kaggle.com/sunnysai12345/news-summary")

        if not os.path.isfile(os.path.expanduser(data_dir_2)):
            raise Exception ("Cannot find" + os.path.expanduser(data_dir_2) + ". If necessary, please download from https://www.kaggle.com/sunnysai12345/news-summary")        
        
        dataset1 = (pd.read_csv(data_dir_1, encoding='iso-8859-1')).iloc[:,0:6].copy()
        dataset2 = (pd.read_csv(data_dir_2, encoding='iso-8859-1')).iloc[:,0:2].copy()
        
        dataset = pd.DataFrame()
        dataset['sentences'] = pd.concat([dataset1['text'], dataset2['text']], ignore_index=True)
        dataset['summaries'] = pd.concat([dataset1['headlines'],dataset2['headlines']],ignore_index = True)
        
        dataset.head(2)
        
        cleaned_sentences = preprocess_utils.text_strip(dataset['sentences'])
        cleaned_summaries = preprocess_utils.text_strip(dataset['summaries'])

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
