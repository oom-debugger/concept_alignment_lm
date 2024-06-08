""" Hierachical community extraction from embedding space.
required packages:
!pip install networkx
!pip install transformers
!pip install umap-learn
!pip install sentencepiece
!pip install torch
"""
import copy
import json
import math
import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import umap

import scipy
import networkx as nx



def get_fuzzy_graph(embeddings, n_neighbors, metric='minkowski', local_connectivity= 1.0):
  (
      graph_,
      _sigmas,
      _rhos,
      graph_dists_,
  ) = umap.umap_.fuzzy_simplicial_set(
      embeddings,
      n_neighbors=n_neighbors,
      random_state=100,
      metric=metric,
      metric_kwds={},
      knn_indices=None,
      knn_dists=None,
      # angular=False,
      angular=True,
      set_op_mix_ratio=1.0,
      local_connectivity=local_connectivity,
      apply_set_operations=True,
      verbose=False,
      return_dists=True)
  G = nx.from_scipy_sparse_array(
      graph_, parallel_edges=True, create_using=nx.Graph
  )
  return G


class Node:
  def __init__(self, rids: list[int], knn: list[int], uid: str, part_strats = None):
    self.uid = uid
    self.knn = knn
    self.rids = rids
    self.strats = part_strats
    if not part_strats:
      if self.knn >= 100:
        self.strats = 'run_leiden'
        # self.strats = 'run_louvain'
      # elif self.rids > 500:
      else:
        self.strats = 'run_louvain'
      # else: #  self.knn >= 70:
      #   self.strats = 'run_girwan'
    self.node_paritions = [] #  list[Node] = []

  def add_child(self, node):
    self.node_paritions.append(node)

  def isleaf(self):
    return False if self.node_partitions else True

def get_h_partitions(rids, knn, strat, embeddings, resolution = 1, threshold = 1e-07, local_connectivity = 1, time_allowed=60):
  G_ = get_fuzzy_graph(embeddings=embeddings[rids], n_neighbors=knn, metric='cosine', local_connectivity=local_connectivity)
  mapping = {gid: rid for gid, rid in zip(sorted(G_), rids)}
  # Get partitions
  if strat == 'run_leiden':
    # leiden
    lv_partitions = nx.community.label_propagation_communities(G_)
  elif strat == 'run_louvain':
    # louvain
    lv_partitions = list(nx.community.louvain_partitions(G_, weight='weight', resolution=resolution, threshold=threshold, seed=None))
    lv_partitions = lv_partitions[1] if len(lv_partitions) > 2 else lv_partitions[0]
  # elif strat == 'run_girwan':
  #   lv_partitions = list(nx.community.girvan_newman(G_))
  else:
    raise ValueError('Unrecognized partitioning strategy!')

  lvs = [sorted(list(p)) for p in lv_partitions]
  lvs.sort(key=len, reverse=True)
  # get real ids for the partitions
  r_lvs = [[mapping[i] for i in lv] for lv in lvs]
  return r_lvs


def dfs_partitioning(cur_node, child_knns, embeddings):
  # resolution = 1, threshold = 1e-07, local_connectivity = 5
  threshold = 0.001
  parts = get_h_partitions(cur_node.rids, cur_node.knn, cur_node.strats, embeddings) # , threshold=threshold)
  for uid, p in enumerate(parts):
    cknn = child_knns[0] if child_knns else 1
    cstrat = 'run_louvain'
    new_child = Node(p, cknn, part_strats=cstrat, uid=f'{cur_node.uid}_{str(uid)}')
    cur_node.add_child(new_child)
    if child_knns:
      dfs_partitioning(new_child, child_knns[1:], embeddings)


def bfs_group_by_knn(root):
  grouped = {}
  queue = [root]
  while queue:
    node = queue.pop(0)
    if node.knn not in grouped:
      grouped[node.knn] = []
    grouped[node.knn].append(node)
    for p in node.node_paritions:
      queue.append(p)
  return grouped

