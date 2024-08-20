
import numpy as np

import torch
from torchmetrics.functional import spearman_corrcoef
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

from transformers import AutoTokenizer
from transformers import AlbertForMaskedLM, T5ForConditionalGeneration



def get_shared_vocab(vocab_1, vocab_2, whitespace_1, whitespace_2):
  # 1. replace whitespaces with <sep>
  # 2. lower case all tokens
  vocab_1 = [v.replace(whitespace_1, '<sep>').lower() for v in vocab_1]
  vocab_2 = [v.replace(whitespace_2, '<sep>').lower() for v in vocab_2]
  # 3. get the intersection and then make a list out of it
  shared_vocab = list(set(vocab_1).intersection(vocab_2))
  # 4. replace the <sep> with whitespace_1/2
  shared_vocab_1 = [tk.replace('<sep>', whitespace_1) for tk in shared_vocab]
  shared_vocab_2 = [tk.replace('<sep>', whitespace_2) for tk in shared_vocab]
  # 5. get token ids
  return (shared_vocab_1, shared_vocab_2)


def calculate_embedding_score(embedding_pool, metric='cosine'):
  if metric == 'cosine':
    return pairwise_cosine_similarity(embedding_pool, zero_diagonal=False)
  elif metric == 'miskowski':
    return torch.cdist(embedding_pool, embedding_pool)


def get_embedding_pool(token_ids, model_name):
  if 't5' in model_name.lower():
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    embeddings = model.shared.weight.detach().cpu()
  elif 'albert' in model_name.lower():
    model = AlbertForMaskedLM.from_pretrained(model_name)
    embeddings = model.albert.embeddings.word_embeddings.weight.detach().cpu()
  else:
    raise ValueError(f'{model_name} not supported!')
  return embeddings[token_ids, :]


def calculated_spearman(model_name_1, model_name_2, whitespace_1, whitespace_2):
  """Only supports Albert and T5"""
  # "google-t5/t5-small" "▁"
  tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1)
  tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
  (shared_vocab_1, 
   shared_vocab_2) = get_shared_vocab(tokenizer_1.get_vocab(), tokenizer_2.get_vocab(), whitespace_1, whitespace_2)
  ids_1 = tokenizer_1.convert_tokens_to_ids(shared_vocab_1)
  ids_2 = tokenizer_2.convert_tokens_to_ids(shared_vocab_2)
  embeddings_1 = get_embedding_pool(token_ids=ids_1, model_name=model_name_1)
  embeddings_2 = get_embedding_pool(token_ids=ids_2, model_name=model_name_2)
  scores_1 = calculate_embedding_score(embeddings_1, metric='cosine')
  scores_2 = calculate_embedding_score(embeddings_2, metric='cosine')
  res = []
  for i in range(scores_1.shape[0]):
    res.append(spearman_corrcoef(scores_1[i], scores_2[i]).item())
  print(f'the spearsman corrleation for the shared tokens  ({len(ids_1)} tokens) is: {np.mean(res)}')