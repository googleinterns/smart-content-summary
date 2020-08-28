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

""" Web app for LaserTagger text summarizer """

from __future__ import print_function

from builtins import FileExistsError
from flask import Flask, render_template, request
import googleapiclient
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer

import bert_example
import bert_example_classifier
import tagging
import tagging_converter
import utils
from predict_utils import construct_example

app = Flask(__name__)

embedding_type = "POS"
label_map_file = "gs://publicly_available_models_yechen/best_hypertuned_POS/label_map.txt"
enable_masking = False
do_lower_case = True

try:
    nltk.download('punkt')
except FileExistsError:
    print("NLTK punkt exist")

try:
    nltk.download('averaged_perceptron_tagger')
except FileExistsError:
    print("NLTK averaged_perceptron_tagger exist")

if embedding_type == "Normal" or embedding_type == "Sentence":
    vocab_file = "gs://lasertagger_training_yechen/cased_L-12_H-768_A-12/vocab.txt"
elif embedding_type == "POS":
    vocab_file = "gs://bert_traning_yechen/trained_bert_uncased/bert_POS/vocab.txt"
elif embedding_type == "POS_concise":
    vocab_file = "gs://bert_traning_yechen/trained_bert_uncased/bert_POS_concise/vocab.txt"
else:
    raise ValueError("Unrecognized embedding type")

label_map = utils.read_label_map(label_map_file)
converter = tagging_converter.TaggingConverter(
    tagging_converter.get_phrase_vocabulary_from_label_map(label_map), True)
id_2_tag = {tag_id: tagging.Tag(tag) for tag, tag_id in label_map.items()}
builder = bert_example.BertExampleBuilder(label_map, vocab_file, 128, do_lower_case,
                                          converter, embedding_type, enable_masking)

grammar_vocab_file = "gs://publicly_available_models_yechen/grammar_checker/vocab.txt"
grammar_builder = bert_example_classifier.BertGrammarExampleBuilder(grammar_vocab_file, 128, False)


def predict_json(project, model, instances, version=None):
    """ Send a json object to GCP deployed model for prediction.
    Args:
      project: name of the project where the model is in
      model: the name of the deployed model
      instances: the json object for model input
      version: the version of the model to use. If not specified,
        will use the default version.

    Returns:
      Inference from the deployed ML model.
    """
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


@app.route('/', methods=['GET'])
def home():
    """Returns the home page of the web app."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """ Receives user inputs, passes to deployed GCP ML models,
    and displays a page with the result."""
    inp_string = [x for x in request.form.values()]
    sentence = nltk.word_tokenize(inp_string[0])
    inputs, example = construct_example(sentence, builder)

    val = predict_json("smart-content-summary", "Deployed_Models", [inputs])
    try:
        predicted_ids = val[0]["pred"]
    except:
        predicted_ids = val[0]
    example.features['labels'] = predicted_ids
    example.features['labels_mask'] = [0] + [1] * (len(predicted_ids) - 2) + [0]
    labels = [id_2_tag[label_id] for label_id in example.get_token_labels()]
    prediction = example.editing_task.realize_output(labels)

    inputs_grammar, example_grammar = construct_example(prediction, grammar_builder)
    grammar_prediction = predict_json("smart-content-summary", "grammar_checker", [inputs_grammar])

    try:
        grammar = grammar_prediction[0]["pred"][0]
    except:
        grammar = grammar_prediction[0][0]

    prediction = TreebankWordDetokenizer().detokenize(prediction.split())
    return render_template('index.html', input=inp_string[0], prediction_bert=prediction, grammar=grammar)


if __name__ == '__main__':
    # For deploying to App Engine
    app.run(host='127.0.0.1', port=8080, debug=True)
    # For local deployment
    # app.run(host='localhost', port=8080, debug=True)
