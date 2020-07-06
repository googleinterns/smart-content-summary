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

from typing import Text, List

import nltk


def convert_to_pos(tokens: List[Text]) -> List[int]:
    """Convert a list of tokens to a list of int representing Part of Speech (POS) tag.
    Args:
        tokens: a list of tokens to be converted.
    Returns:
        tokens_pos: a list of int representing the POS.
    """
    pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS",
                "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$",
                "RB", "RBR", "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG",
                "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", '.', 'X']
    pos_dict = {}
    pos_tag = 3  # saving 0, 1, 2 for other taggings in BERT
    for tag in pos_tags:
        pos_dict[tag] = pos_tag
        pos_tag += 1
    oov_tag = pos_tag

    tokens_pos_tags = nltk.pos_tag(tokens)
    tokens_pos = []
    for row in tokens_pos_tags:
        _, pos = row
        if pos in pos_dict:
            tokens_pos.append(pos_dict[pos])
        else:
            tokens_pos.append(oov_tag)

    return tokens_pos
