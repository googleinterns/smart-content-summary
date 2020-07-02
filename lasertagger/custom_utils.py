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
# limitations under the License

"""Customized utils for modified LaserTagger model."""

import json
import os
import nltk

from typing import Text, List
# nltk.download('averaged_perceptron_tagger')

def write_lasertagger_config(output_dir: Text, bert_type: Text, t2t: bool, number_of_layer: int, hidden_size: int, attention_heads: int, filter_size: int, full_attention: bool):
    """ Write the LaserTagger configuration as a json file.
    
    Args:
        file_dir: the directory where the json file will be stored
        bert_type: the type of Bert. There are two types: Base and POS
        t2t: If True, will use autoregressive decoder. If False, will use feedforward decoder.
        number_of_layer: number of hidden layers in the decoder.
        hidden_size: the size of hidden layer in the decoder.
        attention_heads: number of attention heads in the decoder. 
        filter_size: the size of the decoder filter.
        full_attention: whether to use full attention in the decoder. 
    """
    
    if bert_type == "Base":
        vocab_type = 2
        vocab_size = 28996
    elif bert_type == "POS":
        vocab_type = 42
        vocab_size = 32000
    else:
        raise ValueError("bert_type needs to be 'Base' or 'POS'.")
        
    
    lasertagger_conf = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        'max_position_embeddings': 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": vocab_type,
        "vocab_size": vocab_size,
        "use_t2t_decoder": t2t,
        "decoder_num_hidden_layers": number_of_layer,
        "decoder_hidden_size": hidden_size,
        "decoder_num_attention_heads": attention_heads,
        "decoder_filter_size": filter_size,
        "use_full_attention": full_attention
    }
    
    output_dir = os.path.expanduser(output_dir)

    with open("{}/bert_config.json".format(output_dir), "w") as f:
        json.dump(lasertagger_conf, f, indent=2)


        
def convert_to_POS(tokens: List[Text]) -> List[int]:
    """Convert a list of tokens to a list of int representing Part of Speech (POS) tag.
    Args:
        tokens: a list of tokens to be converted.
    
    Returns:
        tokens_POS: a list of int representing the POS.
    """
    POS_tags = ["CC", "CD", "DT", "EX","FW", "IN", "JJ", "JJR", "JJS", "LS",\
                "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$",\
                "RB", "RBR", "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG",\
                "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", '.', 'X']
    POS_dict = {}
    count = 3 # saving 0, 1, 2 for other taggings in BERT
    for i in POS_tags:
        POS_dict[i] = count
        count += 1
    unidentified_tag = count
    
    toens_POS_tags = nltk.pos_tag(tokens)
    tokens_POS = []
    for row in toens_POS_tags:
        a, b = row
        if b in POS_dict:
            tokens_POS.append(POS_dict[b])
        else:
            tokens_POS.append(unidentified_tag)
    
    return tokens_POS