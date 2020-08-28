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
""" Utils for preprocessing MS dataset."""

GRADING_COMMENTS = ["Most important meaning Flawless language", "Most important meaning Minor errors", \
                    "Most important meaning Disfluent or incomprehensible", "Much meaning Flawless language", \
                    "Much meaning Minor errors", "Much meaning Disfluent or incomprehensible", \
                    "Little or none meaning Flawless language", "Little or none meaning Minor errors", \
                    "Little or none meaning Disfluent or incomprehensible"]
GRADING_NUMBER = ["6", "7", "9", "11", "12", "14", "21", "22", "24"]
EXCLUSION_NUMBER = ["9", "14", "21", "22", "24"]


def process_row(row):
  """Split a row into the original sentence, its corresponding summary and its rating.

    Args:
      row: a row in the MS dataset tsv file.
    Returns:
      current_original_sentence: the original sentence of the row
      current_shortened_sentences_list: a list of summaries corresponding to the current_original_sentence
      current_shortened_ratings_list: the a list of ratings of the summaries in current_shortened_sentences_list
      count_excluded: number of summaries excluded in the row due to low ratings
    """
  count_excluded = 0
  row_flattened = []
  for i in range(len(row)):
    splitted_row = row[i].split(" ||| ")
    for j in range(len(splitted_row)):
      if splitted_row[j] not in GRADING_COMMENTS:
        row_flattened.append(splitted_row[j])

  current_original_sentence = row_flattened[2]
  current_shortened_sentences_list = []
  current_shortened_ratings_list = []

  this_shortened_sentence = row_flattened[3]
  this_shortened_sentence_rating = []
  for i in range(4, len(row_flattened)):
    if i + 1 == len(row_flattened):
      this_shortened_sentence_rating.append(row_flattened[i])
      this_shortened_sentence_rating = this_shortened_sentence_rating[2:]
      if len(this_shortened_sentence_rating) == 0 or \
          len(set(this_shortened_sentence_rating).intersection(set(EXCLUSION_NUMBER))) / \
          len(this_shortened_sentence_rating) < 0.5:
        current_shortened_sentences_list.append(this_shortened_sentence)
        current_shortened_ratings_list.append(this_shortened_sentence_rating)
      else:
        count_excluded += 1

    elif not row_flattened[i].isnumeric() and not row_flattened[i].split(
        ";")[0].isnumeric():
      this_shortened_sentence_rating = this_shortened_sentence_rating[2:]
      if len(this_shortened_sentence_rating) == 0 or \
          len(set(this_shortened_sentence_rating).intersection(set(EXCLUSION_NUMBER))) / \
          len(this_shortened_sentence_rating) < 0.5:
        current_shortened_sentences_list.append(this_shortened_sentence)
        current_shortened_ratings_list.append(this_shortened_sentence_rating)
        this_shortened_sentence = row_flattened[i]
        this_shortened_sentence_rating = []
      else:
        count_excluded += 1
    else:
      this_shortened_sentence_rating.append(row_flattened[i])
  assert (len(current_shortened_sentences_list) == len(
      current_shortened_ratings_list))
  return current_original_sentence, current_shortened_sentences_list, current_shortened_ratings_list, count_excluded
