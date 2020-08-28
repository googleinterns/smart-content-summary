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
""" Post-processing utils for LaserTagger predictions """

from nltk.tokenize.treebank import TreebankWordDetokenizer


def post_processing(input_text):
  """ Removes redundant punctuation marks and detokenize the input string.

    Redundant punctuation marks before, in the middle, and after the sentences
    are removed. For paired punctuation marks (i.e. [], "", '', (), and {}),
    the unpaired ones are removed; if the pair both appear before any words,
    the pair is deleted. The output is detokenized by TreebankWordDetokenizer.

    Args:
      input_text: a string with tokens separated by white space
    Returns:
      A string with redundant punctuation marks removed
    """

  input_text = input_text.split()
  in_punctuation = True
  punctuations = [",", ".", ":", "!", "?"]
  whether_keep_list = []

  left_punctuations = {"{": [], "[": [], "(": [], '"': [], "'": []}
  pairing = {"}": "{", "]": "[", ")": "(", '"': '"', "'": "'"}

  sentence_started = False
  in_punctuation = False

  for index, word in enumerate(input_text):
    if not sentence_started:
      if word in punctuations:
        whether_keep_list.append(False)
      elif word in left_punctuations:
        left_punctuations[word].append(index)
        whether_keep_list.append(True)
      elif word in pairing:
        if len(left_punctuations[pairing[word]]) > 0:
          whether_keep_list[left_punctuations[pairing[word]].pop()] = False
        whether_keep_list.append(False)
      else:
        whether_keep_list.append(True)
        sentence_started = True
        in_punctuation = False
    else:
      if word not in punctuations:
        in_punctuation = False
        whether_keep_list.append(True)

        if word in left_punctuations:
          left_punctuations[word].append(index)

        elif word in pairing:
          if len(left_punctuations[pairing[word]]) == 0:
            whether_keep_list[-1] = False
          else:
            left_punctuations[pairing[word]].pop()

      else:
        if in_punctuation:
          whether_keep_list.append(False)
        else:
          in_punctuation = True
          whether_keep_list.append(True)

  for list_i in left_punctuations.values():
    for i in list_i:
      whether_keep_list[i] = False

  output_text = []
  for i in range(len(whether_keep_list)):
    if whether_keep_list[i]:
      output_text.append(input_text[i])

  return TreebankWordDetokenizer().detokenize(output_text)
