# Summary Quality Classifier

This classifier assesses the quality of text summarization results in terms 
of grammar and meaning preservation, and classify them as acceptable or
unacceptable. This model is modified from the [LaserTagger](
https://github.com/google-research/lasertagger) script, and 
is built on Python 3, Tensorflow and 
[BERT](https://github.com/google-research/bert). 

BERT transformer model is used as the encoder, and for the decoder, a
feed-forward network is applied to the decoder. This model should be trained on
the [Microsoft Abstractive Text Compression Dataset](
https://www.microsoft.com/en-us/download/confirmation.aspx?id=54262).
This dataset contains ratings for each source-summary pair in terms of grammar and 
meaning preservation. For grammar ratings, an additional dataset - [the Corpus of Linguistic Acceptability(CoLA)](
https://nyu-mll.github.io/CoLA/) - can help improve the classification accuracy.

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

python preprocess_MS_dataset_for_classifier.py $INPUT_DIR $NUM_TUNE $NUM_VALID
```

Six tsv files will be saved at your home directory. Three correspond to grammar scoring,
and three are for meaning preservation scoring. 

Note that only 14% of the Microsoft dataset are samples with unacceptable grammar or 
meaningless samples (negative samples). To improve the model performance, it is helpful 
to add more negative samples, either from other dataset or from synthetic data. 
For the grammar classifier, [the Corpus of Linguistic Acceptability(CoLA)](https://nyu-mll.github.io/CoLA/) dataset
can be used. Run the following command to preprocess this dataset and add all its negative samples to the training set for 
the grammar classifier.
 ```
 python preprocess_cola_dataset_for_classifier.py /path/to/CoLA/data /path/to/grammar/training/tsv
 ```
A tsv file named classifier_mixed_training_set_grammar.tsv will be saved, which mixes the Microsoft dataset
with the negative samples from the CoLA dataset. 

To mix the training datasets with other synthetic negative samples to achieve a desirable 
negative sample ratio, you can run 
```
python mixing_in_negative_samples.py path/to/synthetic/negative/sample/file \
                                     path/to/training/data/file \
                                     target_negative_sample_ratio \
                                     path/to/output/file
```

After the preprocessing above, run the following command on all six tsv files (train, tune, 
and validation set for grammar and meaning classifier) to convert the samples to tf_record 
(i.e. change the `INPUT_FILE` to be the path to the train, tune, and validation files, 
change `TYPE_OF_INPUT` to grammar/meaning_train/tune/valid, and change `classifier_type` to 
either Grammar or Meaning respectively). Shown below is an example for 
converting the tuning dataset for grammar classification.
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

To train the classifier for grammar scoring, run the following commands. 
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
  --num_categories=2 \
  --classifier_type=Grammar
```
Similarly, to train the meaning classifier, change the training 
and eval files, output directory, and the classifier type correspondingly.

To export the trained grammar classifier, run the following. 
```
python run_classifier.py \
  --model_config_file=./configs/bert_config.json \
  --output_dir=${OUTPUT_DIR}/grammar_classifier \
  --do_export=true \
  --export_path=${OUTPUT_DIR}/grammar_classifier/export \
  --num_categories=2 \
  --classifier_type=Grammar
```

### 3. Prediction

Run the following to make grammar scoring predictions on the input file. 
```
python predict_main.py \
 --input_file=path/to/input/file \
 --output_file=path/to/output/prediction/file \
 --vocab_file=${BERT_BASE_DIR}/vocab.txt
 --saved_model=path/to/exported/model \
 --batch_size=500 \
 --classifier_type=Grammar
```
To make meaning scoring inferences, change the classifier_type to Meaning.

Compute the accuracy, precision, and recall score of the predictions:
```
python score_main.py path/to/output/prediction/file
```

Example output:
```
Accuracy: 0.8902
----- For category 0 -----
Precision is 0.6846
Recall is 0.4000
F1 is 0.5050
----- For category 1 -----
Precision is 0.9085
Recall is 0.9700
F1 is 0.9383
```
