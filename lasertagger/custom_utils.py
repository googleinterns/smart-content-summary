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

import re
from typing import Text, List

import nltk


def convert_to_pos(tokens: List[Text], pos_type: Text) -> List[int]:
  """Convert a list of tokens to a list of int representing Part of Speech (POS) tag.
    Args:
        tokens: a list of tokens to be converted
        pos_type: type of pos tagging, POS or POS_concise
    Returns:
        tokens_pos: a list of int representing the POS.
    """
  if pos_type == "POS":
    pos_tags = [
        "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
        "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR",
        "RBS", "RP", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
        "WDT", "WP", "WP$", "WRB", '.', 'X'
    ]
    tokens_pos_tags = nltk.pos_tag(tokens)
  elif pos_type == "POS_concise":
    pos_tags = [
        "VERB", "NOUN", "PRON", "ADJ", "ADV", "ADP", "CONJ", "DET", "NUM",
        "PRT", "X", "."
    ]
    tokens_pos_tags = nltk.pos_tag(tokens, tagset='universal')
  else:
    raise ValueError("pos_type must be either POS or POS_concise.")

  pos_dict = {}
  pos_tag = 3  # saving 0, 1, 2 for other taggings in BERT
  for tag in pos_tags:
    pos_dict[tag] = pos_tag
    pos_tag += 1
  oov_tag = pos_tag

  tokens_pos = []
  for row in tokens_pos_tags:
    _, pos = row
    if pos in pos_dict:
      tokens_pos.append(pos_dict[pos])
    else:
      tokens_pos.append(oov_tag)

  return tokens_pos


def prepare_vocab_for_masking(vocab_file):
  """ Add [NUMBER] and [SYMBOL] as the generic masking for numbers and symbols to the
    vocab file if they do not exist.
    
    Args:
      vocab_file: path to the BERT vocab file.
    """
  with open(vocab_file, 'r') as f:
    # read a list of lines into data
    lines = f.readlines()

  lines = __replace_unused_vocab(lines, "[SYMBOL]\n")
  lines = __replace_unused_vocab(lines, "[NUMBER]\n")

  with open(vocab_file, 'w') as file:
    file.writelines(lines)


def __replace_unused_vocab(vocab_list, new_vocab):
  """ Replace unused vocab in the vocab list with new vocab.
    
    Args:
      vocab_list: a list of vocab tokens
      new_vocab: a new vocab that needs to be added to the list
    """
  if new_vocab not in vocab_list:
    pattern = re.compile('\[UNUSED_.*\]\n')
    patter_lowercase = re.compile('\[unused.*\]\n')
    for i, vocab in enumerate(vocab_list):
      if re.match(pattern, vocab) or re.match(patter_lowercase, vocab):
        vocab_list[i] = new_vocab
        return vocab_list
    raise IndexError('There is no unused vocab in the list.')
  return vocab_list
