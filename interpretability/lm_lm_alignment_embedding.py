"""
> python3 interpretability/lm_lm_alignment_embedding.py  \
    --src_model_name=albert-xlarge-v2 \
    --dst_model_name=albert-base-v2  \
    --src_whitespace_char='▁'  \
    --dst_whitespace_char='▁' 
"""
import os
import copy
import json
import numpy as np
import pprint
import torch

from collections import defaultdict
from torchmetrics.functional import spearman_corrcoef
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from transformers import AutoTokenizer
from transformers import AlbertForMaskedLM, T5ForConditionalGeneration, AutoModelForCausalLM, LlamaForCausalLM


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-src_model_name", "--src_model_name", dest = "src_model_name", default = None, help="first model name (huggingface).", type=str, required=True)
parser.add_argument("-dst_model_name", "--dst_model_name", dest = "dst_model_name", default = None, help="second model name (huggingface).", type=str, required=True)
parser.add_argument("-src_whitespace_char", "--src_whitespace_char", dest = "src_whitespace_char", default = None, type=str, required=True,
                    help="Source whitespace characters to remove")
parser.add_argument("-dst_whitespace_char", "--dst_whitespace_char", dest = "dst_whitespace_char", default = None, type=str, required=True,
                    help="Source whitespace characters to remove")
parser.add_argument("-thresholds", "--thresholds", dest = "thresholds", default = '3,5,10,50,100', type=str,
                    help="top-k thresholds for the spearsman correlation evaluation.")
parser.add_argument("-input_dir", "--input_dir", dest = "input_dir", default = None, help="input dir for sentences")

def get_valid_words(input_dir):
    # !wget 'https://zenodo.org/record/5172857/files/wiki_morph.json'
    # !wget 'https://enroots.neocities.org/families.txt'

    with open(os.path.join(input_dir, 'families.txt'), 'r') as f:
        data = f.readlines()
    # read word families
    world_families_dict = defaultdict(list)
    world_families_set = set()
    for d in data:
      if d.startswith('\t'):
        world_families_dict[fam].append(d.strip())
        world_families_set.add(d.strip())
      else:
        fam = d.strip()
        world_families_set.add(fam)
    # read Morph Wiki
    with open(os.path.join(input_dir, 'wiki_morph.json'), 'r') as f:
      morph_train_dataset = json.load(f)
    morph_train_dataset[0]['Word'], morph_train_dataset[0]['PoS']
    morph_vocab = [v['Word'] for v in morph_train_dataset]
    word_morph_dict = defaultdict(lambda: defaultdict(list))
    for entry in morph_train_dataset:
      word = entry['Word']
      pos = entry['PoS']
      word_morph_dict[word][pos].append(entry)
    morph_vocabs = list(word_morph_dict.keys())
    return set(morph_vocabs).union(world_families_set)
    
def get_shared_word_vocab(vocab_1, vocab_2, whitespace_1, whitespace_2, valid_words):
  # 1. replace whitespaces with <sep>
  # 2. lower case all tokens
  # vocab_1 = [v.replace(whitespace_1, '<sep>').lower() for v in vocab_1 if not keep_only_whitespace or v.startswith(whitespace_1)]
  # vocab_2 = [v.replace(whitespace_2, '<sep>').lower() for v in vocab_2 if not keep_only_whitespace or v.startswith(whitespace_2)]
  vocab_1 = [v.replace(whitespace_1, '') for v in vocab_1 if v.startswith(whitespace_1)]
  vocab_2 = [v.replace(whitespace_2, '') for v in vocab_2 if v.startswith(whitespace_2)]
  # 3. get the intersection and then make a list out of it
  shared_vocab = sorted(list(set(vocab_1).intersection(vocab_2)))
  shared_valid_vocab = list(set(shared_vocab).intersection(valid_words))

  # 4. replace the <sep> with whitespace_1/2
  shared_vocab_1 = [whitespace_1 + tk for tk in shared_valid_vocab if tk.isalpha()]
  shared_vocab_2 = [whitespace_2 + tk for tk in shared_valid_vocab if tk.isalpha()]
  # 5. get token ids
  return (shared_vocab_1, shared_vocab_2)


def get_shared_vocab(vocab_1, vocab_2, whitespace_1, whitespace_2, keep_only_whitespace):
  # 1. replace whitespaces with <sep>
  # 2. lower case all tokens
  # vocab_1 = [v.replace(whitespace_1, '<sep>').lower() for v in vocab_1 if not keep_only_whitespace or v.startswith(whitespace_1)]
  # vocab_2 = [v.replace(whitespace_2, '<sep>').lower() for v in vocab_2 if not keep_only_whitespace or v.startswith(whitespace_2)]
  vocab_1 = [v.replace(whitespace_1, '<sep>') for v in vocab_1 if not keep_only_whitespace or v.startswith(whitespace_1)]
  vocab_2 = [v.replace(whitespace_2, '<sep>') for v in vocab_2 if not keep_only_whitespace or v.startswith(whitespace_2)]
  # 3. get the intersection and then make a list out of it
  shared_vocab = sorted(list(set(vocab_1).intersection(vocab_2)))
  # 4. replace the <sep> with whitespace_1/2
  shared_vocab_1 = [tk.replace('<sep>', whitespace_1) for tk in shared_vocab]
  shared_vocab_2 = [tk.replace('<sep>', whitespace_2) for tk in shared_vocab]
  # 5. get token ids
  return (shared_vocab_1, shared_vocab_2)


def calculate_embedding_score(embedding_pool, metric='cosine'):
  if metric == 'cosine':
    return pairwise_cosine_similarity(embedding_pool, zero_diagonal=True)
  elif metric == 'miskowski':
    return torch.cdist(embedding_pool, embedding_pool)



def get_embedding_pool(token_ids, model_name):
  if 't5' in model_name.lower():
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    embeddings = model.shared.weight.detach().cpu()
  elif 'albert' in model_name.lower():
    model = AlbertForMaskedLM.from_pretrained(model_name)
    embeddings = model.albert.embeddings.word_embeddings.weight.detach().cpu()
  elif 'gemma' in model_name.lower():
    # "google/gemma-2b"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    embeddings = copy.deepcopy(model.model.embed_tokens.weight.detach().cpu().to(torch.float16))
    del model
  elif 'llama' in model_name.lower():
    # "llama"
    model = LlamaForCausalLM.from_pretrained(model_name)
    embeddings = model.model.embed_tokens.weight.detach().cpu().to(torch.float16)
  else:
    raise ValueError(f'{model_name} not supported!')
  return embeddings[token_ids, :]

   
def calculated_global_scores(tokenizer, model_name, shared_vocab, metric):
  ids = tokenizer.convert_tokens_to_ids(shared_vocab)
  embeddings = get_embedding_pool(token_ids=ids, model_name=model_name)
  scores = calculate_embedding_score(embeddings, metric=metric)
  del embeddings
  return scores


def get_sorted(distance, metric, max_k):
  return torch.argsort(distance, dim=-1, stable=True, descending=True if metric == 'cosine' else False)[:, :max_k]


def calculated_global_spearman(scores_1, scores_2):
  res = []
  for i in range(scores_1.shape[0]):
    res.append(spearman_corrcoef(scores_1[i], scores_2[i]).item())
  print(f'the spearsman corrleation for the shared tokens  ({scores_1.shape[0]} tokens) is: {np.mean(res)}')


def calculated_top_k_scores(
    model_name_1, model_name_2,
    whitespace_1, whitespace_2,
    keep_only_whitespace,
    k_lst, metric='cosine',
    input_dir=None):
  tokenizer_base = AutoTokenizer.from_pretrained(model_name_1)
  tokenizer_l = AutoTokenizer.from_pretrained(model_name_2)
  if not input_dir:
      (shared_vocab_base, 
        shared_vocab_l) = get_shared_vocab(
            tokenizer_base.get_vocab(), tokenizer_l.get_vocab(), 
            whitespace_1, whitespace_2, keep_only_whitespace)
  else:
      (shared_vocab_base, 
        shared_vocab_l) = get_shared_word_vocab(
            tokenizer_base.get_vocab(), tokenizer_l.get_vocab(), 
            whitespace_1, whitespace_2,
            valid_words=get_valid_words(input_dir),
        )
  max_k = max(k_lst)
  score_base = calculated_global_scores(tokenizer_base, model_name_1, shared_vocab_base, metric=metric)
  # score_base = calculated_cosine_scores_mem_efficient(tokenizer_base, model_name_1, shared_vocab_base, metric=metric, max_k=max_k)
  sorted_index_base = get_sorted(score_base, metric=metric, max_k=max_k)
  print ('get first pairwise similarity....', sorted_index_base.shape)

  scores_l = calculated_global_scores(tokenizer_l, model_name_2, shared_vocab_l, metric=metric)
  # scores_l = calculated_cosine_scores_mem_efficient(tokenizer_l, model_name_2, shared_vocab_l, metric=metric, max_k=max_k)
  sorted_index_l = get_sorted(scores_l, metric=metric, max_k=max_k)
  print ('get second pairwise similarity....', sorted_index_base.shape)

  for k in k_lst:
    accs = []
    for xid in range(len(shared_vocab_l)):
      if not keep_only_whitespace or shared_vocab_base[xid].startswith(whitespace_1):
        accs.append(len(set(sorted_index_l[xid][:k].tolist()).intersection(set(sorted_index_base[xid][:k].tolist()))) / k)
    print(f'For k:{k} mean acc is:{np.mean(accs)}. Shared vocab size:{len(shared_vocab_l)}')



def main():
  args = parser.parse_args()
  calculated_top_k_scores(
    model_name_1=args.src_model_name, 
    model_name_2=args.dst_model_name,  
    whitespace_1=args.src_whitespace_char, 
    whitespace_2=args.dst_whitespace_char, 
    keep_only_whitespace=False,
    metric='cosine',
    k_lst=[int(k) for k in args.thresholds.split(',')],
    input_dir=args.input_dir,
    )


if __name__ == "__main__":
    main()
