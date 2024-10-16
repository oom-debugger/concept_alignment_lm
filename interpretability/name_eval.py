"""Add evaluation for human-eval alignment.

!pip -q install transformers
!pip -q install sentencepiece
!pip -q install names-dataset

> python3 interpretability/name_eval.py \
    --src_cluster=<your dir>/clusters/llama-3.1-70b_clusters.json \
    --src_whitespace_char='Ġ'  \
    --layer_names=6,12,25  \
    --model='meta-llama/Llama-3.1-70B'
"""

import json
from transformers import AutoTokenizer
from names_dataset import NameDataset
nd = NameDataset()

import argparse
from huggingface_hub import login

parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model", dest = "model", default = None, help="model name")
parser.add_argument("-src_cluster", "--src_cluster", dest = "src_cluster", default = None, help="Source cluster set", type=str, required=True)
parser.add_argument("-src_whitespace_char", "--src_whitespace_char", dest = "src_whitespace_char", default = None, type=str, required=True,
                    help="Source whitespace characters to remove")
parser.add_argument("-layer_names", "--layer_names", dest = "layer_names", default = '75,50,25,12', type=str,
                    help="list of knns to be evaluated.")
parser.add_argument("-gender_confidence", "--gender_confidence", dest = "gender_confidence", default = 0.7, help="gender_confidence when evaluating first names.", type=bool)
parser.add_argument("-min_cluster_size", "--min_cluster_size", dest = "min_cluster_size", default = 15, help="gender_confidence when evaluating first names.", type=int)

############################################### Libraries #########################################
################ Get all clusters that contains first names ################
def get_countries(cluster, first_name, tokenizer, whitespace = '▁'):
  countries = set()
  for name in cluster:
    name = get_decoded(name, tokenizer).replace(whitespace, '').lower()
    res = nd.search(name)
    try:
      res = res['first_name'] if first_name else res['last_name']
      prob = set(res['rank'].keys())
      countries.update(prob)
    except:
      pass
  return countries


def get_first_name_ids(raw_wordpieces, tokenizer, rank_th= 1000, gender_confidence = 0.95, first_name=True, whitespace = '▁'):
  name_ids = []
  c = 0
  for id, tk in enumerate(raw_wordpieces):
    if tk.startswith(whitespace):
      name = get_decoded(tk, tokenizer).replace(whitespace, '').lower()
      res = nd.search(name)
      if first_name:
        res = res['first_name']
      else:
        res = res['last_name']
      if res and 'gender' in res and len(res['gender']):
        gender = sorted(res['gender'].items(), key=lambda item: item[1], reverse=True)[0] if first_name else ('N/A', 1.0)
        rank = get_best_rank(name, res['rank'])
        if gender[1] > gender_confidence and rank[1] and rank[1] < rank_th:
          name_ids.append((id, tk, gender))
          c += 1
  print (f'number of detected {"first" if first_name else "last"} names: ', c)
  return name_ids


def get_last_name_ids(raw_wordpieces, tokenizer, rank_th= 1000, gender_confidence = 0.95, first_name=False, whitespace = '▁'):
  name_ids = []
  c = 0
  for id, tk in enumerate(raw_wordpieces):
    name = get_decoded(tk, tokenizer).replace(whitespace, '').lower()
    if len(name) > 3:
      res = nd.search(name)
      res = res['first_name'] if first_name else res['last_name']
      if res:
        rank = get_best_rank(name, res['rank'])
        if rank[1] and rank[1] < rank_th:
          name_ids.append((id, tk, 'N/A'))
          c += 1
  print (f'number of detected {"first" if first_name else "last"} names: ', c)
  return name_ids


def get_name_clusters(h_clusters, layer_names, flat_name_uids, tokenizer, evaluated_clusters, whitespace = '▁'):
  # layer_names = ['100', '75', '50', '25']
  first_name_clusters = {}
  for k in sorted([int(i) for i in layer_names], reverse=True):
    for c in h_clusters[str(k)]:
        if c['uid'] in flat_name_uids:
          if is_evaluated(c['uid'], evaluated_clusters):
            continue
          print (c['uid'], len(c['tokens']), [get_decoded(i, tokenizer).replace(whitespace, '').lower() for i in c['tokens']])
          first_name_clusters[c['uid']] = [get_decoded(i, tokenizer).replace(whitespace, '').lower() for i in c['tokens']]
  return first_name_clusters


def get_flat_name_uids(h_clusters, layer_names, tokenizer, min_cluster_size, whitespace, first_name,  gender_confidence,  rank_th,  cluster_th):
#   layer_names = [ '75', '50', '25']
  name_uids = {}
  flat_name_uids = set()
  for knn in sorted([int(i) for i in layer_names]):
    knn = str(knn)
    name_uids[knn] = set()
    # 1. make sure the smaller clusters don't be subset of what we already added.
    for c in h_clusters[knn]:
      already_in = False
      for ft_id in flat_name_uids:
        if c['uid'].startswith(ft_id):
          already_in = True
          break
      if already_in:
        continue
      # see if the cluster is name cluster
      count = 0
      for tk in c['tokens']:
        name = get_decoded(tk, tokenizer).replace(whitespace, '').lower()
        if name.isalpha():
          res = nd.search(name)
          res = res['first_name'] if first_name else res['last_name']
          if res and (not first_name or ('gender' in res and len(res['gender']))):
            gender = sorted(res['gender'].items(), key=lambda item: item[1], reverse=True)[0] if first_name else ('N/A', 1.0)
            rank = get_best_rank(name, res['rank'])
            if (not first_name or gender[1] > gender_confidence) and rank[1] and rank[1] < rank_th:
              count += 1
        if (count / len(c['tokens'])) >= cluster_th and len(c['tokens']) >= min_cluster_size:
              name_uids[knn].add(c['uid'])
              flat_name_uids.add(c['uid'])
  return (flat_name_uids)


def eval_c(cluster, country, first_name, rank_th, gender_confidence, tokenizer, whitespace = '▁'):
  genders = []
  probs = []
  count = 0
  for tk in cluster:
    # if tk.startswith('▁'):
    name = get_decoded(tk, tokenizer).replace(whitespace, '').lower()
    res = nd.search(name)
    try:
      res = res['first_name'] if first_name else res['last_name']
      gender = sorted(res['gender'].items(), key=lambda item: item[1], reverse=True)[0] if first_name else ('N/A', 1.0)
      prob = res['country'][country] if 'rank' in res else 0.0
      rank = res['rank'][country] if 'rank' in res else 100_000
    except:
      prob = 0.0
      gender = ('N/A', 0.01)
      rank = 100_000
    if (gender_confidence == 0 or gender[1] > gender_confidence) and rank and rank < rank_th:
      # name_ids.append((id, tk, gender))
      count += 1
      genders.append(gender)
      probs.append(prob)
  if len(genders) > 0:
    div =  len(genders)
    female_score = sum([1 if g[0].lower() == 'female' else 0 for g in genders]) /div
    male_score = sum([1 if g[0].lower() == 'male' else 0 for g in genders]) / div
    is_female = 'female' if female_score > male_score else 'male'
  else:
    is_female = 'no-gender'
  return is_female, probs, count / len(cluster)


def eval_latino_countries(h_clusters, country_dict, country_code_mapping, tokenizer, whitespace = '▁'):
  # Evaluating the latino cluster names. Note that since the latin city names tend to be multi-words, we calculate the matching
  # different way.

  def belonging(word, lst):
    for l in lst:
      if l.startswith(word + ' ') or l.endswith(' ' + word):
        return True
    return False
  latin_countries = {
      'argentina',
      'bolivia',
      'brazil',
      'chile',
      'colombia',
      'cuba',
      'ecuador',
      'guatemala',
      'haiti',
      'honduras',
      'mexico',
      'nicaragua',
      'panama',
      'paraguay',
      'peru',
      'philippines',
      'spain',
      'uruguay',
      'venezuela'}
  latin_names = set()
  for i in latin_countries:
    if 'philippines' != i:
      ns = nd.get_top_names(n=1000, country_alpha2=country_code_mapping[i])
      for n in ns.values():
        latin_names.update(n)

  latin_loc = set()
  for i in latin_countries:
    latin_loc.update(country_dict[i])

  for c in h_clusters['75']:
    if c['uid'].startswith('0_1_6'):
      n_l = latin_names.union(latin_loc)
      tp_count = 0
      for i in c['tokens']:
        if belonging(get_decoded(i, tokenizer).replace(whitespace, '').lower(), n_l):
          tp_count += 1
      print('overall precision for latino/hispanic NERs:', tp_count / len(c['tokens']))
  return


def get_best_rank(name, ref):
  # ref = res['first_name']['rank']
  if ref:
    ref = {k: v if v else 100_000 for k, v in ref.items()}
    rank = sorted(ref.items(), key=lambda item: item[1])[0]
  else:
    rank = ('None-Country', 100_000)
  return rank


def is_evaluated(uid, evaluated_clusters):
  for cluster in evaluated_clusters:
    if uid.startswith(cluster + '_') or uid == cluster:
      return True
  return False


def evaluate_names(src_cluster, ref_names, layer_names, tokenizer, whitespace, first_name=True, rank_th=1000, gender_confidence=0.5, cluster_th = 0.7, min_cluster_size= 8, evaluated_clusters = None):
  flat_name_uids= get_flat_name_uids(
    src_cluster, tokenizer=tokenizer, layer_names=layer_names, whitespace=whitespace,
    min_cluster_size= min_cluster_size, first_name=first_name, 
    gender_confidence=gender_confidence,  rank_th=rank_th,  cluster_th=cluster_th)
  if not evaluated_clusters:
    evaluated_clusters = []
  first_name_clusters = get_name_clusters(src_cluster, tokenizer=tokenizer, layer_names=layer_names, flat_name_uids=flat_name_uids, 
                                          whitespace=whitespace, evaluated_clusters=evaluated_clusters)

  # last_name_clusters
  for name, cluster in first_name_clusters.items():
    if is_evaluated(name, evaluated_clusters):
      continue
    if len(cluster) >= min_cluster_size:
      precisions = []
      countries = []
      for country in get_countries(cluster, first_name=True, tokenizer=tokenizer):
        is_female, _, precision = eval_c(cluster, tokenizer=tokenizer, country=country, first_name=first_name, rank_th=rank_th, gender_confidence=gender_confidence, whitespace=whitespace)
        precisions.append(precision)
        countries.append(country)
      val = len(set(cluster).intersection(ref_names)) / len(set(cluster))
      if val > 0.5: # cluster_th 
        print('-------------------------------------------------------------------------------------------------------------------')
        # print (len(cluster), cluster)
        print ('name_pre:', val, len(cluster))
        print (name, is_female, sorted([(c, round(p, 3)) for c, p in zip(countries, precisions)], key=lambda item: item[1], reverse=True))
      evaluated_clusters.append(name)
  return evaluated_clusters
###################################################################################################################################################

# def get_decoded(tk, tokenizer=None):
_VOCAB_CACHE = {}
def get_decoded(tk, tokenizer=None):
  if tk in _VOCAB_CACHE:
    return _VOCAB_CACHE[tk]
  elif tokenizer:
    res = tokenizer.decode(tokenizer.vocab[tk])
    _VOCAB_CACHE[tk] = res
    tk = res
  return tk
  
def make_wordpiece_from_a_layer(src_cluster):
  cluster_layer = list(src_cluster.values())[0]
  tokens = []
  for c in cluster_layer:
    tokens.extend(c['tokens'])
  return tokens


def main():
  args = parser.parse_args()
  with open(args.src_cluster) as f:
    src_cluster = json.load(f)
  whitespace = args.src_whitespace_char
  layer_names = sorted(args.layer_names.split(','))
  tokenizer = None
  if args.model:
    tokenizer = AutoTokenizer.from_pretrained(args.model)

  vocab = make_wordpiece_from_a_layer(src_cluster)
  print('================================================================================================')
  first_names =  get_first_name_ids(vocab, tokenizer=tokenizer, rank_th= 1000, gender_confidence = args.gender_confidence, whitespace=whitespace)
  fnames = set(l[1] for l in first_names if len(l[1]) >= 4)
  last_names =  get_last_name_ids(vocab, tokenizer=tokenizer, rank_th= 1000, gender_confidence = args.gender_confidence, first_name=False, whitespace=whitespace)
  lnames = set(l[1] for l in last_names if len(l[1]) >= 4)
  names = fnames.union(lnames)
  ref_names = set(get_decoded(c, tokenizer).replace(whitespace, '').lower() for c in names)
  ref_lnames = set(get_decoded(c, tokenizer).replace(whitespace, '').lower() for c in lnames)
  print('=====================================First Name Eval Numbers=============================================')
  ref_fnames = set(get_decoded(c, tokenizer).replace(whitespace, '').lower() for c in fnames)
  evaluated_clusters = evaluate_names(
    src_cluster, ref_fnames, layer_names, tokenizer, whitespace, 
    first_name=True, rank_th=1000, gender_confidence=args.gender_confidence, cluster_th = 0.5, min_cluster_size=args.min_cluster_size)
  print('================================================================================================') 
  # 0_0_0_0_0_3
  print('=====================================Last Name Eval Numbers=============================================')
  evaluate_names(
    src_cluster, ref_lnames, layer_names, tokenizer, whitespace,
    first_name=False, rank_th=1000, gender_confidence=0, cluster_th = 0.5, min_cluster_size=args.min_cluster_size, 
    evaluated_clusters=evaluated_clusters)
  print('================================================================================================')


  # surname_uids = {}
  # flat_surname_uids = set()
  # # The user need to mention the top level cluster in order to avoid all clusters for name.
  # flat_surname_uids.add('0_1_0')

  # for knn in ['100', '75', '50', '25']:
  #   surname_uids[knn] = set()
  #   # 1. make sure the smaller clusters don't be subset of what we already added.
  #   for c in src_cluster[knn]:
  #     already_in = False
  #     for ft_id in flat_surname_uids:
  #       if c['uid'].startswith(ft_id):
  #         already_in = True
  #         break
  #     if already_in:
  #       continue
  #     # see if the cluster is name cluster
  #     count = 0
  #     for tk in c['tokens']:
  #       # if tk.startswith('▁'):
  #       name = get_decoded(tk, tokenizer).replace(whitespace, '').lower()
  #       if name.isalpha() and len(name) > 3:
  #         res = nd.search(name)
  #         res = res['first_name'] if first_name else res['last_name']
  #         if res:
  #           rank = get_best_rank(name, res['rank'])
  #           if rank[1] and rank[1] < rank_th:
  #             # name_ids.append((id, tk, gender))
  #             count += 1
  #       if (count / len(c['tokens'])) >= cluster_th and len(c['tokens']) > 7:
  #             surname_uids[knn].add(c['uid'])
  #             flat_surname_uids.add(c['uid'])

  # flat_surname_uids = flat_surname_uids - flat_name_uids
  # len(flat_surname_uids)
  # print('================================================================================================')

  # last_name_clusters = {}
  # for c in src_cluster['100']:
  #   if c['uid'] in flat_surname_uids:
  #     print (c['uid'], len(c['tokens']), [get_decoded(i, tokenizer).replace(whitespace, '').lower() for i in c['tokens']])
  #     last_name_clusters[c['uid']] = [get_decoded(i, tokenizer).replace(whitespace, '').lower() for i in c['tokens']]
  # for c in src_cluster['75']:
  #   if c['uid'] in flat_surname_uids:
  #     print (c['uid'], len(c['tokens']), [get_decoded(i, tokenizer).replace(whitespace, '').lower() for i in c['tokens']])
  #     last_name_clusters[c['uid']] = [get_decoded(i, tokenizer).replace(whitespace, '').lower() for i in c['tokens']]
  # for c in src_cluster['50']:
  #   if c['uid'] in flat_surname_uids:
  #     print (c['uid'], len(c['tokens']), [get_decoded(i, tokenizer).replace(whitespace, '').lower() for i in c['tokens']])
  #     last_name_clusters[c['uid']] = [get_decoded(i, tokenizer).replace(whitespace, '').lower() for i in c['tokens']]
  # for c in src_cluster['25']:
  #   if c['uid'] in flat_surname_uids:
  #     print (c['uid'], len(c['tokens']), [get_decoded(i, tokenizer).replace(whitespace, '').lower() for i in c['tokens']])
  #     last_name_clusters[c['uid']] = [get_decoded(i, tokenizer).replace(whitespace, '').lower() for i in c['tokens']]

  # _lnames = set(get_decoded(c, tokenizer).replace(whitespace, '').lower() for c in lnames)
  # for name, cluster in last_name_clusters.items():
  #   precisions = []
  #   countries = []
  #   if len(cluster) >= 5:
  #     for country in get_countries(cluster, first_name=False, tokenizer=tokenizer):
  #       _, _, precision = eval_c(cluster, tokenizer=tokenizer, country=country, first_name=False, rank_th=3000, gender_confidence=0.0, whitespace=whitespace)
  #       precisions.append(precision)
  #       countries.append(country)
  #     print('-------------------------------------------------------------------------------------------------------------------')
  #     print ('cluster size:', len(cluster))
  #     print ('name_pre:', len(set(cluster).intersection(_lnames)) / len(set(cluster)))
  #     print (name, sorted([(c, round(p, 3)) for c, p in zip(countries, precisions)], key=lambda item: item[1], reverse=True))

if __name__ == "__main__":
    main()
