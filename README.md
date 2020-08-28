# Text Summarization based on LaserTagger

Based on a text-editing model called [LaserTagger](
https://github.com/google-research/lasertagger), this project aims to train 
a machine learning model that rewrites short sentences and phrases to a more 
concise form. 

The application of this project includes:
- Creative suggestion: When the creatives provided by the customers or the 
suggested creatives from existing models exceed word limit, this model can 
provide an automatic summary of the text.
- Keyword/category clustering: This model can generate a shorter version for 
each category and keyword, and cluster those with the same shortened version 
together. 

The LaserTagger model is developed and trained by Google Research, which 
transforms a source text into a target text by predicting a sequence of 
token-level edit operations. The goal of this project is to improve the 
performance of this model specifically for the short-sentence-and-phrase 
summarization task. 

Our improvement of the model includes adding part-of-speech (POS) tags to 
BERT embeddings (which involves pre-training BERT with POS tags), 
customizing loss function and loss weights, applying masks, and 
hyperparameter tuning. 

To address the lack of grammar evaluation in existing performance metrics 
for text summarization, we designed and trained a grammar [checker](classifier). We 
provide the code and instructions for training the grammar checker.

The end-to-end process of this model can be deployed on Google Cloud 
Platform, where the web interface accepts a text input, and returns its 
summarized version along with a grammar rating of the summary. The 
[code](GCP_deploy) for the deployment is also provided.

## Modified LaserTagger
The modified LaserTagger is built on Python 3, Tensorflow and BERT. It works 
with CPU, GPU, and Cloud TPU. In addition to improving the model performance, 
we also provide code to streamline the training and exporting process, and 
making  predictions faster by running inferences in batches.

The LaserTagger model uses BERT as the encoder. There are pre-trained BERT 
models online, but adding part-of-speech (POS) tags to the embeddings involves 
retraining the BERT model. We provide two pretrained BERT models trained on 
the [OpenSubtitles](https://www.opensubtitles.org/en/search/subs) dataset. The 
BERT model with POS tags can be found at 
gs://bert_traning_yechen/trained_bert_uncased/bert_POS. The BERT model with 
POS-concise tags can be found at 
gs://bert_traning_yechen/trained_bert_uncased/bert_POS_concise. If you plan to 
use "Normal" embeddings or "Sentence" embeddings which do not include POS tags, 
you can download a pretrained BERT model from the [official repository](
https://github.com/google-research/bert#pre-trained-models). You can use 
either the 12-layer ''BERT-Base, Cased'' model or the 12-layer ''BERT-Base, 
Uncased'' model. 

### Usage Instructions

**1. Data Preprocessing**

The dataset we train the model on is the 
[Microsoft Abstractive Text Compression Dataset](
https://www.microsoft.com/en-us/download/confirmation.aspx?id=54262) (MSF dataset). 
To preprocess the dataset, and split to train, tune, and test set, run the 
following command

```
python preprocess_MS_dataset_main.py path/to/raw/data num_of_tuning num_of_testing
```
where we use 3,000 samples for tuning and 3,000 samples for testing in the project. 
The preprocessed and split dataset will be saved in three tsv files named 
train_MS_dataset, tune_MS_dataset, and tune_MS_dataset for training, tuning, and 
testing set respectively.

We also provide preprocessing scripts for three other datasets: 
[news summary dataset](https://www.kaggle.com/sunnysai12345/news-summary), 
[Google sentence compression dataset](
https://github.com/google-research-datasets/sentence-compression), and 
[reddit_tifu dataset](https://www.tensorflow.org/datasets/catalog/reddit_tifu). 
See [preprocess_news_dataset_main.py](preprocess_news_dataset_main.py), 
[preprocess_google_dataset_main.py](preprocess_google_dataset_main.py), and 
[preprocess_reddit_dataset_main.py](preprocess_reddit_dataset_main.py) for code and instructions.

The preprocessing script also computes basic statistics of the dataset. A 
sample output when preprocessing the MSF dataset is
```
Number of samples is 26119
Total number of excluded sample is 304
Average word count of original sentence is 32.08 ( std: 10.79 )
Max word count is 145
Min word count is 7
Average word count of shortened sentence is 22.28 ( std: 8.54 )
Max Length is 108
Min Length is 3
On average, there are 1.00 sentences in each original text ( std: 0.00 )
On average, there are 1.91 words in each shortened sentence that are not in the original sentence. ( std: 3.76 )
The average compression ratio is 0.70 ( std: 0.19 )
```

**2. Training & Export**

This streamlined process covers the steps of phrase vocabulary optimization, 
preparing data for training, model training, and model export. The script is 
designed for training on the Google Cloud Platform. Before running the script, 
there are several prerequisites:
- Create or have access to a Google Storage Bucket. Currently, the GCP bucket 
path is set to gs://trained_models_yechen/. If you create another bucket, change 
the path by changing the GCP_BUCKET variable in [streamline_training.py](streamline_training.py).
-  Set up a virtual machine to run the script on. Follow this [guide](
https://docs.google.com/document/d/1oV8Swp_BDfmDHkhSkWb2wo_ZhC9jIP-Lk7kCbYvdYTM/edit#heading=h.o18hkt51hrci) 
to set up a virtual machine on GCP.
- If you plan to train with a Cloud TPU, follow this [guide](
https://docs.google.com/document/d/1PlCB6DOH8LUBsN8UcgPxzds9MqPRFIPs_rht9fWqbAA/edit?usp=sharing) 
to set up a TPU on GCP. Make sure that your TPU has the same name as your VM. 
- Download the pre-trained BERT model from sources suggested above. To copy a 
folder in the GCP bucket to your virtual machine, use the `gsutil cp` command.

After satisfying the prerequisites, you can run the streamline script 
[streamline_training.py](streamline_training.py). The usage is 
```
python streamline_training.py \
[-vocab_size VOCAB_SIZE] [-train_batch_size TRAIN_BATCH_SIZE] \
[-learning_rate LEARNING_RATE] [-num_train_epochs NUM_TRAIN_EPOCHS] \
[-warmup_proportion WARMUP_PROPORTION] \
[-max_input_examples MAX_INPUT_EXAMPLES] \
[-train] [-export] \
[-use_tpu] [-gbucket GBUCKET] \
[-t2t T2T] [-number_layer NUMBER_LAYER] \
[-hidden_size HIDDEN_SIZE] [-num_attention_head NUM_ATTENTION_HEAD] \
[-filter_size FILTER_SIZE] [-full_attention FULL_ATTENTION] \
[-add_tag_loss_weight ADD_TAG_LOSS_WEIGHT] \
[-delete_tag_loss_weight DELETE_TAG_LOSS_WEIGHT] \
[-keep_tag_loss_weight KEEP_TAG_LOSS_WEIGHT] \
model/output/dir abs/path/to/lasertagger abs/path/to/bert \
path/to/training/file path/to/tuning/file \
embedding_type
```
The positional arguments are:
- `model/output/dir`: the directory of the model output
- `abs/path/to/lasertagger`: absolute path to the folder where the lasertagger 
scripts are located
- `abs/path/to/bert`: absolute path to the folder where the pretrained BERT is 
located
- `path/to/training/file`: path to training samples
- `path/to/tuning/file`: path to tuning samples
- `embedding_type`: type of embedding. Must be one of [Normal, POS, POS_
concise, Sentence]. Normal: segment id is all zero. POS: part of speech tagging. 
POS_concise: POS tagging with a smaller set of tags. Sentence: sentence tagging.

The general optional arguments are:
- `-train`: if added, skip preprocessing and start training.
- `-export`: if added, skip preprocessing and training, and start exporting to 
bucket.

The optional arguments relevant to the data preprocessing step are:
- `-vocab_size VOCAB_SIZE`: the size of the vocabulary for the adding tag. 
default = 500
- `-max_input_examples MAX_INPUT_EXAMPLES`: number of training examples to use 
in the vocab optimization. default is all training data
- `-masking`: if added, numbers and symbols will be masked. All numbers are 
replaced with the [NUMBER] token, and all special characters other than ., !, 
?, ;, and , are replaced with the [SYMBOL] token.

The optional arguments relevant to the training step are:
- `-train_batch_size TRAIN_BATCH_SIZE`: batch size during training. default 
= 32
- `-learning_rate LEARNING_RATE`: the initial learning rate for Adam. default 
= 3e-5
- `-num_train_epochs NUM_TRAIN_EPOCHS`: total number of training epochs to perform. 
default = 3
- `-warmup_proportion WARMUP_PROPORTION`: proportion of training to perform linear 
learning rate warmup for. default = 0.1
- `-use_tpu`: if added, will use cloud TPU for training.
- `-gbucket GBUCKET`: the gcp bucket where the cloud TPU will store intermediary 
outputs to.
- `-verb_deletion_loss VERB_DELETION_LOSS`: the weight of verb deletion loss. Need 
to be >= 0. default=0. Cannot be set to a number other than 0 unless the 
embedding_type is POS or POS_concise.
`-add_tag_loss_weight ADD_TAG_LOSS_WEIGHT`: the weight of loss for adding tags. default=1
`-delete_tag_loss_weight DELETE_TAG_LOSS_WEIGHT`: the weight of loss for deleting tags. default=1
`-keep_tag_loss_weight KEEP_TAG_LOSS_WEIGHT`: the weight of loss for keeping tags. default=1

The optional arguments relevant to the model architecture are:
- `-t2t T2T`: if True, use autoregressive version of LaserTagger. If false, use, 
feed-forward version of LaserTagger. default = True
- `-number_layer NUMBER_LAYER`: number of hidden layers in the decoder. default
= 1
- `-hidden_size HIDDEN_SIZE`: the size of the hidden layer size in the decoder. 
default=768
- `-num_attention_head NUM_ATTENTION_HEAD`: the number of attention heads in the 
decoder. default=4
- `-filter_size FILTER_SIZE`: the size of the filter in the decoder. default = 
3072
- `-full_attention FULL_ATTENTION`: whether to use full attention in the decoder. 
default = False

The trained and exported model will be saved at the local directory specified by 
`model/output/dir` and in the GCP bucket in a folder whose name is the last folder 
name of model/output/dir. Currently, the GCP bucket is set to be 
gs://trained_models_yechen/. If you would like to save to another bucket, change 
the GCP_BUCKET variable in [streamline_training.py](streamline_training.py). 

**3.  Prediction**

The [custom_predict.py](custom_predict.py) runs prediction on input and computes SARI score and exact 
score if applicable. The usage is
```
python custom_predict.py [-score] \ 
path/to/input/file path/to/lasertagger path/to/bert \
models [models ...] \
embedding_type
```

The positional arguments are:

- `path/to/input/file`: the path to the tsv file with inputs. If the scores do not 
need to be computed, the tsv should have one column which contains the inputs. If 
scores need to be computed, then the tsv file needs to have two columns, with the 
first column being the inputs and the second column being the targets.
- `path/to/lasertagger`: path to the folder where the lasertagger scripts are located
- `path/to/bert`: path to the folder where the pretrained BERT is located
abs_path_to_lasertagger
- `models`: the name of trained LaserTagger models. Need to provided at least one 
model name. The model name should be the folder name of the LaserTagger model in the 
GCP bucket.
- `embedding_type`: type of embedding. Must be one of [Normal, POS, POS_concise, 
Sentence].

optional arguments:
- `-score`: if added, compute scores for the predictions.
- `-grammar`: if added, automatically apply grammar correction on predictions 
using LanguageTool.
- `-masking`: if added, numbers and symbols will be masked.
- `-batch_size`: the batch size of prediction. default=1

If you add the `-grammar` tag for automatic grammar correction, you need to install 
the LanguageTool using following commands:
```
pip install 3to2
sudo apt update
sudo apt install default-jre
pip install language-tool-python
```
All predictions are written to a tsv file named pred.tsv. The first column is the 
original input. The last column is the targets if targets are provided. All other 
columns are predictions from different models specified by the `models` arguments. 
If the `-score` tag is added, another tsv file named score.tsv will also be generated. 
This file contains six rows. The first row is the model names; the second row is the 
exact scores; the third row is the SARI score; the fourth to sixth row are the keep, 
addition, and delete scores (of the SARI score). Each column corresponds to the scores 
of a model. 

## Grammar & Meaning Preservation Checker
There is a lack of grammar evaluations in the existing text summarization metrics. 
Therefore, we design a model that can classify a text as grammatically correct or 
incorrect. Follow the instructions in [classifier](classifier) to preprocess the MSF 
dataset, train the model, and make predictions. This model can also be used for 
checking whether a summary preserves the most important meaning in the source text. 

## Deployment to Google Cloud Platform
The LaserTagger model can be deployed on GCP with a web interface. The process 
involves three main steps: 1. exporting the LaserTagger to a format acceptable to the 
GCP AI platform, 2. deploying the model to AI platform, and 3. deploying the web 
application to the GCP App Engine.

### 1. Re-export the LaserTagger Model
The GCP app engine only accepts exported model with one metagraph. However, if the 
model is trained on GPU or TPU, the exported version from above very likely contains 
more than one metagraph. To re-export the model, run the 
[export_model_for_gcp.py](GCP_deploy/export_model_for_gcp.py) in 
the [GCP_deploy](GCP_deploy) folder.
```
python run the export_model_for_gcp.py exported/model/dir output/dir
```
The `exported/model/dir` is the path to a folder containing an exported LaserTagger 
model. The `output/dir` specified where you would like the re-exported model to be 
saved. After re-exporting the model, check that the model indeed only has one 
metagraph by running the following command:
```
saved_model_cli show --dir=output/dir --all
```
You should only see one `signature_def['serving_default']` in the output. 

### 2. Deploy to GCP AI Platform
First, use the `gsutil cp` command to copy the exported LaserTagger model to a GCP 
bucket. Then, create a model on the AI platform by running:
```
export MODEL_NAME=lasertagger
gcloud ai-platform models create $MODEL_NAME --enable-logging
```
Then, run the following commands to deploy the exported model as a new version:
```
export MODEL_DIR=path/to/model/in/GCP/bucket
export VERSION_NAME=version_name
export FRAMEWORK=TENSORFLOW
gcloud beta ai-platform versions create $VERSION_NAME \
--model $MODEL_NAME \
--origin $MODEL_DIR \
--runtime-version=1.15 \
--framework $FRAMEWORK --python-version=3.7 \
--accelerator=count=4,type=nvidia-tesla-v100 \
--machine-type=n1-standard-32
```
Later, we can send our inputs as a json object to this deployed model, and 
inferences will be made by the AI platform on a virtual machine with 4 GPUs, which 
can significantly decrease the inference time.

### 3. Build the Web Application on GCP App Engine
Open a shell terminal in the GCP console. Use `git clone` to copy the repository to 
the shell terminal. In addition, copy the following files and folders in the 
[lasertagger](lasertagger) folder to the gcp_deploy folder: [bert](lasertagger/bert),
[bert_example.py](lasertagger/bert_example.py), [custom_utils.py](lasertagger/custom_utils.py), 
[tagging.py](lasertagger/tagging.py), [tagging_converter.py](lasertagger/tagging_converter.py), 
and [utils.py](lasertagger/utils.py). Then running the following to deploy the web app:
```
cd gcp_deploy
gcloud app deploy
```
This deployment will take a few minutes. After it is successfully built, run 
`gcloud app browse` to find the link to the web page. 
## License

Apache 2.0; see [LICENSE](LICENSE) for details.

## Disclaimer

**This is not an officially supported Google product.**
