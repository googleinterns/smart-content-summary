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
nltk.download('averaged_perceptron_tagger')


        
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
