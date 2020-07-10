# Summary Quality Classifier

This classifier assesses the quality of text summarization result in terms 
of grammar and meaning preservation. This model modified from the 
[LaserTagger](https://github.com/google-research/lasertagger) script, and 
is built on Python 3, Tensorflow and 
[BERT](https://github.com/google-research/bert). 

BERT transformer model is used as the encoder, and for the decoder, a single
feed-forward pass is applied to the decoder. This model should be trained on
the 
[Microsoft Abstractive Text Compression Dataset](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54262).
This dataset contains ratings for each source-summary pair, and is scored on
grammar and meaning on a 0-2 scale. 

## Usage Instructions

A standard three-step procedure (preprocessing, training, and evaluation)
is needed for an experiment with this model.

### 1. Preprocessing

Download the 
[Microsoft Abstractive Text Compression Dataset](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54262)
and unzip. You should find a folder named "RawData". Run the following to 
preprocess the dataset. 

```
export NUM_TUNE=500
export NUM_VALID=500
export INPUT_DIR=/path/to/RawData

python preprocess_MS_dataset_for_classifier.py INPUT_DIR NUM_TUNE NUM_VALID
```

Six tsv files will be saved at your home directory. Three corresponding to grammaring scoring,
and three for meaning preservation scoring. Next, run the following command 
on all six tsv files to convert the samples to tf_record (i.e. change the `INPUT_FILE`
to the path to training, tuning, and validation files, change `TYPE_OF_INPUT` to 
grammar/meaning_train/tune/valid, and change `classifier_type` to either 
Grammar or Meaning respectively).

```
export INPUT_FILE=/path/to/input/file
export TYPE_OF_INPUT=grammar_tune 
export OUTPUT_DIR=/path/to/output
export BERT_BASE_DIR=/path/to/pretrained/bert
python preprocess_main.py \
  --input_file=$INPUT_FILE \
  --output_tfrecord=${OUTPUT_DIR}/${TYPE_OF_INPUT}.tf_record \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --classifier_type Grammar
```

### 2. Training


### 3. Prediction

