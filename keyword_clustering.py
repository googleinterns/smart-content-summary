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
"""Cluster keywords based on the result of summarization."""

import argparse
import csv
import os


def cluster_keywords(input_file_path):
  """ Cluster keywords based on the shorted version of the keywords read from the input file.
  Args:
    input_file_path: the path to the tsv file containing keywords and their shortened version.
  Returns:
    shortened_keywords_list: a dictionary with shortened keywords as key, and list of keywords in that cluster as value
    total_keyword_counts: total number of keywords being clustered 
    model_name_list: a list of names of the LaserTagger models used for shortening keywords, as indicated in tsv file
  """
  shortened_keywords_list = []
  total_keyword_counts = 0

  with open(input_file_path) as f:
    read_tsv = csv.reader(f, delimiter="\t")
    model_name_list = next(read_tsv)[1:]
    for i in range(len(model_name_list)):
      shortened_keywords_list.append({})

    for line in read_tsv:
      total_keyword_counts += 1
      for index, shortened_keyword in enumerate(line[1:]):
        shortened_keyword = shortened_keyword.lower()
        if shortened_keyword == "":
          continue
        if shortened_keyword not in shortened_keywords_list[index]:
          shortened_keywords_list[index][shortened_keyword] = [line[0]]
        else:
          shortened_keywords_list[index][shortened_keyword].append(line[0])
  return shortened_keywords_list, total_keyword_counts, model_name_list


def main(args):
  """ Cluster keywords based on their shortened version from the LaserTagger model.

  Reads keywords and their shortened version from tsv files, and write out the clustering results to a new tsv file.

  Args:
    args: command line arguments
  Raises:
    ValueError when input file cannot be found
  """
  input_file_path = os.path.expanduser(args.input_file_path)
  if not os.path.isfile(input_file_path):
    raise ValueError("Cannot find the input file.")

  shortened_keywords_list, total_keyword_counts, model_name_list = cluster_keywords(
      input_file_path)

  output_file_path = args.output_file_path
  file_name = output_file_path.split("/")[-1].split(".")[0]

  for index, model in enumerate(model_name_list):
    clustered_keywords = 0
    print(model)
    if len(model_name_list) == 1:
      model = ""
    this_output_file_path = "/".join(output_file_path.split("/")[:-1]) + "/" + \
                            file_name + "_" + model + ".tsv"

    with open(os.path.expanduser(this_output_file_path), 'wt') as f:
      tsv_writer = csv.writer(f, delimiter='\t')
      for shortened_keywords, keyword_list in shortened_keywords_list[
          index].items():
        if len(keyword_list) > 1:
          clustered_keywords += len(keyword_list)
          keyword_list = [shortened_keywords, len(keyword_list)] + keyword_list
          tsv_writer.writerow(keyword_list)

    print(clustered_keywords, "of", total_keyword_counts,
          "keywords are clustered")


if __name__ == "__main__":
  """Cluster keywords based on their outputs in the LaserTagger model.
    
    usage: keyword_clustering.py [-h] input_file_path output_file_path

    positional arguments:
      input_file_path   the directory of the LaserTagger model output
      output_file_path  Where the clustering result will be stored. If there is output from only one model 
      in the input file, then the output will be saved at this path. Otherwise, the name of the model will be 
      appended to the end of the file name and the clustering result corresponding to each model will be 
      saved in separate files.
    
    The input file format:
    The input file needs to be a tsv file should contain two or more columns. The first column should be
    the original keyword, and the second and later columns should be the corresponding shortened versions
    of the keyword output by different models. The first row of the second and later columns needs to be 
    the name of the model.
    """
  parser = argparse.ArgumentParser()
  parser.add_argument("input_file_path",
                      help="the directory of the LaserTagger model output")
  parser.add_argument(
      "output_file_path",
      help="Where the clustering result will be stored. If there "
      "is output from only one model in the input file, then the output will be saved"
      "at this path. Otherwise, the name of the model will be appended to the end "
      "of the file name and the clustering result corresponding to each model will "
      "be saved in separate files. ")

  arguments = parser.parse_args()

  if arguments.output_file_path[-4:] != ".tsv":
    raise ValueError("The output file path should be .tsv")

  main(arguments)
