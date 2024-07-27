""" Add the evaluation for LM-LM alignment."""
import json
from collections import Counter


def token_glove_mapping(cluster_layer):
  token_mapping = {}
  for id, cluster in enumerate(cluster_layer):
    for tk in cluster['tokens']:
      token_mapping[tk] = cluster['uid']
  return token_mapping


def cluster_percentage(ref_layer, token_mapping, th, debug = False):
  same_count = 0
  loose_count = 0
  vloose_count = 0
  skip_cluster = 0
  for id, ref_cluster in enumerate(ref_layer):
    cnt_lbl = []
    length = 0
    for tk in ref_cluster['tokens']:
      if '▁' in tk:
        tk = tk.replace('▁', '')
        length += 1
        if tk in token_mapping:
          cnt_lbl.append(token_mapping[tk])
    if not cnt_lbl or len(cnt_lbl) == 1:
      skip_cluster += 1
    else:
      # strict count
      counts = Counter(cnt_lbl)
      ratio = counts.most_common()[0][1] /  length  if length != 1 else 1.0
      if ratio >= th:
        same_count += 1
      # Loose count
      lcounts = Counter([s[:s.rindex('_')] for s in cnt_lbl])
      lratio = lcounts.most_common()[0][1] /  length  if length != 1 else 1.0
      if lratio >= th:
        loose_count += 1
      elif debug:
        print('------------------------------------------')
        print(ref_cluster['tokens'])
        print([s[:s.rindex('_')] for s in cnt_lbl])
      # Very Loose Count
      try:
        vlcounts = Counter([s[:s[:s.rindex('_')].rindex('_')] for s in cnt_lbl])
        vlratio = vlcounts.most_common()[0][1] /  length  if length != 1 else 1.0
        if vlratio >= th:
          vloose_count += 1
      except:
        pass
  return (
      round(same_count / (len(ref_layer) - skip_cluster), 3),
      round(loose_count / (len(ref_layer) - skip_cluster), 3),
      round(vloose_count / (len(ref_layer) - skip_cluster), 3),
      skip_cluster,
  )


def main():
  args = parser.parse_args()
  try:
    with open('albert_cluster.json') as f:
        albert_clusters = json.load(f)
    with open('glove_clusters.json') as f:
        ref_clusters = json.load(f) 
  except:
    raise ValueError('you must already have the ALBERT clusters and glove clusters in the working directory as json file.')
  
  for k in ['75', '50', '25' '12', '6']:
    ref_mapping = token_glove_mapping(ref_clusters[k])
    c = {}
    for th in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
      c[str(th)] = cluster_percentage(albert_clusters['50'], ref_mapping, th)
      print(f'k: {k}, th: {th}, length of cluster: {len(albert_clusters['50'])},  value: {c[str(th)]}')



if __name__ == "__main__":
    main()
