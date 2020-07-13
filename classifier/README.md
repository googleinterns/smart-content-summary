# Summary Quality Classifier

This classifier assesses the quality of text summarization result in terms 
of grammar and meaning preservation. This model is modified from the 
[LaserTagger](https://github.com/google-research/lasertagger) script, and 
is built on Python 3, Tensorflow and 
[BERT](https://github.com/google-research/bert). 

BERT transformer model is used as the encoder, and for the decoder, a
feed-forward network is applied to the decoder. This model should be trained on
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

Six tsv files will be saved at your home directory. Three correspond to grammar scoring,
and three are for meaning preservation scoring. Next, run the following command 
on all six tsv files to convert the samples to tf_record (i.e. change the `INPUT_FILE`
to the path to the training, tuning, and validation files, change `TYPE_OF_INPUT` to 
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

To train the classifier for Grammar scoring, run the following commands. 
```
python run_classifier.py \
  --training_file=${OUTPUT_DIR}/train_grammar.tf_record \
  --eval_file=${OUTPUT_DIR}/tune_grammar.tf_record \
  --model_config_file=./configs/bert_config.json \
  --output_dir=${OUTPUT_DIR}/grammar_classifier \
  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  --do_train=true \
  --do_eval=true \
  --train_batch_size=32 \
  --save_checkpoints_steps=500 \
  --num_train_examples=25120 \
  --num_eval_examples=500 \
  --num_categories=3 \
  --classifier_type=Grammar
```
Similarly, to train the classifier for Meaning scoring, change the training 
and eval files, output directory, and the classifier type correspondingly.

To export the trained Grammar model, run the following. 
```
python run_classifier.py \
  --model_config_file=./configs/bert_config.json \
  --output_dir=${OUTPUT_DIR}/grammar_classifier \
  --do_export=true \
  --export_path=${OUTPUT_DIR}/grammar_classifier/export \
  --num_categories=3 \
  --classifier_type=Meaning
```

### 3. Prediction

Run the following to make Grammar scoring predictions on the input file. 
```
! python predict_main.py \
 --input_file=path/to/input/file \
 --output_file=path/to/output/file \
 --vocab_file=${BERT_BASE_DIR}/vocab.txt
 --saved_model=path/to/exported/model \
 --batch_size=500 \
 --classifier_type=Grammar
```
To make Meaning scoring inferences, change the classifier_type to Meaning.
