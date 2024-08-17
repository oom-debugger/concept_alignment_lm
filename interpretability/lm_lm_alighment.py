""" Add the evaluation for LM-LM alignment."""
from typing import Set
import json
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-src_cluster", "--src_cluster", dest = "src_cluster", default = None, help="Source cluster set", type=str, required=True)
parser.add_argument("-dst_cluster", "--dst_cluster", dest = "dst_cluster", default = None, help="destinatoin cluster set", type=str, required=True)
parser.add_argument("-src_whitespace_char", "--src_whitespace_char", dest = "src_whitespace_char", default = None, type=str, required=True,
                    help="Source whitespace characters to remove")
parser.add_argument("-dest_whitespace_char", "--dest_whitespace_char", dest = "dest_whitespace_char", default = None, type=str, required=True,
                    help="Source whitespace characters to remove")
parser.add_argument("-thresholds", "--thresholds", dest = "thresholds", default = '50,60,70,80,90', type=str,
                    help="comma separate list of thresholds for the evaluation.")


def get_token_cluster_map(cluster_layer, whitespace: str = ''):
  """Get the mapping between the tokens and the cluster name.
  
  args:
    cluster_layer: A cluster set that is extracted given a k.

  returns:
    a mapping where the keys are tokens and the values are cluster names.
  """
  token_mapping = {}
  for cluster in cluster_layer:
    for tk in cluster['tokens']:
      if not whitespace or tk.startswith(whitespace):
        token_mapping[tk.replace(whitespace, '')] = cluster['uid']
  return token_mapping, 


def get_cluster_map(cluster_layer, whitespace: str = ''):
  # '▁'
  cluster_map = {}
  for cluster in cluster_layer:
    cluster_map[cluster['uid']] = set([t.replace(whitespace, '') for t in cluster['tokens'] if (not whitespace or t.startswith(whitespace))])
  return cluster_map


def get_max_scores(src_cluster_map, dest_cluster_map, dest_token_map):
  """Given a token cluster from source set, find the max similairy scores.
  """
  precisions = []
  f_scores = []
  for src_cluster in src_cluster_map.values():
    dest_id, prec_tp = Counter([dest_token_map[t] for t in src_cluster]).most_common()[0]
    precisions.append(prec_tp / len(src_cluster) if len(src_cluster) > 0 else 0)
    # if int(max(precisions) >= threshold):
    #   if strict_mapping:
    tp = src_cluster.intersection(dest_cluster_map[dest_id])
    fp = src_cluster - dest_cluster_map[dest_id]
    fn =  dest_cluster_map[dest_id] - src_cluster
    f_scores.append(tp / (tp + 0.5 * (fp + fn)) if (tp + 0.5 * (fp + fn)) > 0.0001 else 0)
  return precisions, f_scores
      


def get_mapping_score(
    src_cluster_map, dest_token_map, dest_cluster_map, threshold):
  p_scores = []
  f_scores = []
  for p, f in get_max_scores(src_cluster_map, dest_cluster_map, dest_token_map):
     p_scores.append(1 if p >= threshold else 0)
     f_scores.append(1 if f >= threshold else 0)
  p_mapping_score = sum(p_scores) / len(p_mapping_score) if len(p_mapping_score) else 0
  f_mapping_score = sum(f_scores) / len(f_mapping_score) if len(f_mapping_score) else 0
  return p_mapping_score, f_mapping_score


def main():
  args = parser.parse_args()
  with open(args.src_cluster) as f:
    src_cluster = json.load(f)
  with open(args.dest_cluster) as f:
    dest_cluster = json.load(f)
  
  for k in ['75', '50', '25' '12', '6']:
    src_cluster_map = get_cluster_map(cluster_layer=src_cluster[k], whitespace=args.src_whitespace_char)
    dest_cluster_map = get_cluster_map(cluster_layer=dest_cluster[k], whitespace=args.dest_whitespace_char)
    dest_token_map = get_token_cluster_map(cluster_layer=dest_cluster[k])
    c = {}
    for th in args.thresholds.split(','):
      c[str(th)] = get_mapping_score(
        src_cluster_map=src_cluster_map[k], 
        dest_token_map=dest_token_map, 
        dest_cluster_map=dest_cluster_map[k], 
        threshold=th)
      print(
        f'k: {k}, th: {th}  p_value: {c[str(th)][0]}  f_value: {c[str(th)][1]}'
        f'size of src cluster: {len(src_cluster_map[k])}, size of src cluster: {len(src_cluster_map[k])}'
        )


if __name__ == "__main__":
    main()
