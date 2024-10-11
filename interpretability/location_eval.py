"""Add evaluation for human-eval alignment.

!pip -q install transformers
!pip -q install sentencepiece
!pip -q install names-dataset
# !pip -q install umap-learn
!pip -q install Unidecode
!git clone 'https://github.com/dr5hn/countries-states-cities-database.git'


> python3 interpretability/location_eval.py \
    --src_cluster=/home/mehrdad/clusters/albert-xlarge-v2_clusters.json \
    --src_whitespace_char='▁'  \
    --location_file='/home/mehrdad/git/countries-states-cities-database/json/countries+states+cities.json' \
    --layer_names=12,25,50,75

> python3 interpretability/location_eval.py \
    --src_cluster=/home/mehrdad/clusters/llama-3.1-70b_clusters.json \
    --src_whitespace_char='Ġ'  \
    --location_file='/home/mehrdad/git/countries-states-cities-database/json/countries+states+cities.json' \
    --layer_names=6,12,25  \
    --model='meta-llama/Llama-3.1-70B'
"""

import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from transformers import AutoTokenizer
import numpy as np
from unidecode import unidecode
import json

import argparse
from huggingface_hub import login

parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model", dest = "model", default = None, help="model name")
parser.add_argument(
  "-location_file", "--location_file", dest = "location_file", default = None,
  help="first model name (huggingface).", type=str, required=True)
parser.add_argument("-src_cluster", "--src_cluster", dest = "src_cluster", default = None, help="Source cluster set", type=str, required=True)
parser.add_argument("-src_whitespace_char", "--src_whitespace_char", dest = "src_whitespace_char", default = None, type=str, required=True,
                    help="Source whitespace characters to remove")
parser.add_argument("-layer_names", "--layer_names", dest = "layer_names", default = '75,50,25,12', type=str,
                    help="list of knns to be evaluated.")
############################################### Libraries #########################################
################################ Location ################################
def get_location_dataset(location_file, use_unidecode = True):
    """
    args:
      location_file:
        json file that contains the hierachical information for the world's location. must be
        compatible with 'https://github.com/dr5hn/countries-states-cities-database.git' format.
      use_unidecode:
        decodes to unicode. needs to be for most of the tokenization schemes.
    """
    with open(location_file, 'r') as f:
      data = json.load(f)
    lst_countries = set()
    lst_states = set()
    lst_cities = set()
    country_dict = {}
    for country in data:
      cnt = unidecode(country['name'].lower()) if use_unidecode else country['name'].lower()
      lst_countries.add(cnt)
      country_dict[cnt] = set()
      country_dict[cnt].add(country['nationality'])
      for state in country['states']:
          lst_states.add(unidecode(state['name'].lower()))
          country_dict[cnt].add(unidecode(state['name'].lower()))
          for city in state['cities']:
            lst_cities.add(unidecode(city['name'].lower()))
            country_dict[cnt].add(unidecode(city['name'].lower()))
    all_places = lst_countries.union(lst_states).union(lst_cities)
    print('number of countries in reference dataset: ', len(lst_countries))
    print('number of states in reference dataset: ', len(lst_states))
    print('number of cities in reference dataset: ', len(lst_cities))
    print('number of all places in reference dataset: ', len(all_places))
    country_code_mapping = {}
    for country in data:
        cnt = unidecode(country['name'].lower()) if use_unidecode else country['name'].lower()
        country_code_mapping[cnt] = country['iso2']
    sub_rejoin = {}
    for country in data:
      subregion_nm = unidecode(country['region'].lower())
      if subregion_nm not in sub_rejoin:
        sub_rejoin[subregion_nm] = set()
      cnt = unidecode(country['name'].lower()) if use_unidecode else country['name'].lower()
      sub_rejoin[subregion_nm].add(cnt)
      cptl = unidecode(country['capital'].lower()) if use_unidecode else country['capital'].lower()
      sub_rejoin[subregion_nm].add(cptl)
      ntlt = unidecode(country['nationality'].lower()) if use_unidecode else country['nationality'].lower()
      sub_rejoin[subregion_nm].add(ntlt)
    return lst_countries, lst_states, lst_cities, all_places, country_code_mapping, country_dict, sub_rejoin


def calculate_precision(cluster, ref, tokenizer, th = 0.5, whitespace = '▁'):
  def get_decoded(tk):
    if tk in _VOCAB_CACHE:
      return _VOCAB_CACHE[tk]
    elif tokenizer:
      res = tokenizer.decode(tokenizer.vocab[tk])
      _VOCAB_CACHE[tk] = res
      tk = res
    return tk
  tokens = [get_decoded(tk).replace(whitespace, '').replace(' ', '').lower() for tk in cluster if get_decoded(tk).startswith(whitespace)]
  # tokens = [get_decoded(tk).replace(whitespace, '').replace(' ', '').lower() for tk in cluster if get_decoded(tk).startswith(whitespace)]
  precision = 0
  if tokens:
    precision = len(set(tokens).intersection(set(ref))) / len(tokens)
  return precision, len(tokens)


def all_location_eval(h_clusters, layer_names, locations, th, tokenizer, min_cluster_size = 3, whitespace = '▁'):
  print ('==================== all location eval =========================')
  skip_names = set()
  ###########################
  def is_subset(uid):
    for l in skip_names:
      if uid.startswith(l):
        return True
    return False
  ###########################
  def eval_layer(cluster_layer):
    for i, c in enumerate(cluster_layer):
      if is_subset(c['uid']) or len(c['tokens']) < min_cluster_size:
        continue
      precision, len_tk = calculate_precision(c['tokens'], ref=locations, tokenizer=tokenizer, th=th, whitespace=whitespace)
      if precision >= th and len_tk >= min_cluster_size:
        skip_names.add(c['uid'])
        print(c['uid'], len(c['tokens']), precision, c['tokens'])
        # print(c['uid'], len_tk, precision)
  ###########################
  for k in sorted([int(i) for i in layer_names], reverse=True):
    print (f'----------------------cluster_layer:{k}-----------------------')
    eval_layer(h_clusters[str(k)])
  return skip_names


# Can be used for both country_dict as well as sub_rejoin dict
def eval_selected_cluster(h_clusters, layer_names, cluster_names, country_dict,  tokenizer, th=0.5, whitespace = '▁'):
  print ('==================== Country eval =========================')
  ###########################
  def eval_layer(cluster_layer):
    for i, c in enumerate(cluster_layer):
      if c['uid'] not in cluster_names:
        continue
      for country_name, locations in country_dict.items():
        precision, len_tk = calculate_precision(c['tokens'], ref=locations, tokenizer=tokenizer, th=th, whitespace=whitespace)
        if precision >= th:
          print(c['uid'], country_name, len_tk, precision) # , c['tokens'])
  ###########################
  for k in sorted([int(i) for i in layer_names], reverse=True):
    print (f'----------------------cluster_layer:{k}-----------------------')
    eval_layer(h_clusters[str(k)])
  return


_VOCAB_CACHE = {}
def main():
  args = parser.parse_args()
  with open(args.src_cluster) as f:
    src_cluster = json.load(f)
  
  # layer_names = ['75', '50', '25', '12', '6']
  layer_names = args.layer_names.split(',')
  tokenizer = None
  if args.model:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
  lst_countries, lst_states, lst_cities, all_places, country_code_mapping, country_dict, sub_rejoin = get_location_dataset(args.location_file)
  # Evaluate countries
  included_cluster_names = all_location_eval(
    src_cluster, layer_names, locations=all_places, 
    tokenizer=tokenizer,
    th=0.5, min_cluster_size=10, whitespace=args.src_whitespace_char)
  # Can be used for both country_dict as well as sub_rejoin dict
  eval_selected_cluster(
    src_cluster, layer_names, included_cluster_names, country_dict,
    tokenizer=tokenizer,
    th=0.5, whitespace = args.src_whitespace_char)



if __name__ == "__main__":
    main()