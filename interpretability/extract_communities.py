""" API invocatoins to extract clusters for ALBERT, T5, and GloVE.

>  python3 interpretability/extract_communities.py  --model="albert-xxlarge-v2"  --output_dir=/home/mehrdad/clusters
"""
import os
import copy
import warnings 

import torch.nn
from transformers import  AlbertTokenizer, AlbertModel, AutoModelForCausalLM
from transformers import AutoTokenizer 
from transformers import MT5Model, T5Tokenizer, T5Tokenizer, T5EncoderModel
from huggingface_hub import notebook_login
# from torchtext.vocab import GloVe, vocab
from concept_extraction_lib import cluster_and_store

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model", dest = "model", default = "albert-xxlarge-v2", help="model name")
parser.add_argument("-output_dir", "--output_dir", dest = "output_dir", default = None, required=True, help="output path for the cluster file.")


# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 

# def get_hg_sp_model(model_name, knns = [125, 100, 75, 50, 25, 12, 6]):
#   """get Gemma clusters. You need to login and authenticate your account on HuggingFace."""
#   tokenizer = AutoTokenizer.from_pretrained(model_name)
#   # raw_word_pieces = set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids)
#   vocab_size = len(tokenizer.get_vocab().values())
#   raw_wordpieces = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
#   model = AutoModel.from_pretrained(model_name)


def get_Gemma(model_name="google/gemma-2b", out_dir = './' , knns = [125, 100, 75, 50, 25, 12, 6]):
  """get Gemma clusters. You need to login and authenticate your account on HuggingFace."""
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  raw_wordpieces = []
  for i in range(len(tokenizer.vocab.values())):
    raw_wordpieces.append(tokenizer._convert_id_to_token(i))
  model = AutoModelForCausalLM.from_pretrained(model_name)
  gemma_embeddings = copy.deepcopy(model._modules['model']._modules['embed_tokens'].weight.detach().cpu())
  print('Running the hierarchical cluster for %s tokens for Gemma Model...' % gemma_embeddings.shape[0])
  output_file_path = os.path.join(out_dir, f'{model_name}_clusters.json')
  if os.path.exists(output_file_path):
      raise ValueError('Cannot overwrite the output dataset!')
  os.makedirs(out_dir, mode = 0o777, exist_ok = True) 
  cluster_and_store(
    embeddings=gemma_embeddings, 
    wordpieces=raw_wordpieces,
    knns=knns,
    output_file_path=output_file_path,
  )

def get_t5(model_name="google-t5/t5-small", out_dir = './', knns = [125, 100, 75, 50, 25, 12, 6]):
  """get T5 clusters. You need to login and authenticate your account on HuggingFace."""

  tokenizer = T5Tokenizer.from_pretrained(model_name)
  raw_wordpieces = tokenizer.sp_model.id_to_piece(list(range(32000)))
  # raw_wordpieces = [t.lower() for t in raw_wordpieces]
  t5_model = T5EncoderModel.from_pretrained(model_name)
  embeddings = t5_model._modules['shared'].weight.detach().cpu()
  print('Running the hierarchical cluster for %s tokens for T5 Model...' % embeddings.shape[0])
  output_file_path = os.path.join(out_dir, f'{model_name}_clusters.json')
  if os.path.exists(output_file_path):
      raise ValueError('Cannot overwrite the output dataset!')
  os.makedirs(out_dir, mode = 0o777, exist_ok = True) 
  cluster_and_store(
    embeddings=embeddings, 
    wordpieces=raw_wordpieces,
    knns=knns,
    output_file_path=output_file_path,
  )

def get_mt5(model_name="google/mt5-small", out_dir = './' , knns = [125, 100, 75, 50, 25, 12, 6]):
  """get T5 clusters. You need to login and authenticate your account on HuggingFace."""
  tokenizer = T5Tokenizer.from_pretrained(model_name)
  raw_wordpieces = tokenizer.sp_model.id_to_piece(list(range(32000)))
  model = MT5Model.from_pretrained(model_name)
  embeddings = model._modules['shared'].weight.detach().cpu()
  print('Running the hierarchical cluster for %s tokens for mT5 Model...' % embeddings.shape[0])
  output_file_path = os.path.join(out_dir, f'{model_name}_clusters.json')
  if os.path.exists(output_file_path):
      raise ValueError('Cannot overwrite the output dataset!')
  os.makedirs(out_dir, mode = 0o777, exist_ok = True) 
  cluster_and_store(
    embeddings=embeddings, 
    wordpieces=raw_wordpieces,
    knns=knns,
    output_file_path=output_file_path,
  )


def get_albert(model_name = 'albert-xxlarge-v2', out_dir = './' ,knns = [125, 100, 75, 50, 25, 12, 6]):
  """get ALBERT clusters. You need to login and authenticate your account on HuggingFace."""
  tokenizer = AlbertTokenizer.from_pretrained(model_name)  # ALBERT-xxlarge
  raw_wordpieces = tokenizer.sp_model.id_to_piece(list(range(0, 30000)))
  model = AlbertModel.from_pretrained(model_name)
  embeddings = model._modules['embeddings']._modules['word_embeddings'].weight.detach().cpu()
  print('Running the hierarchical cluster for %s tokens for Albert Model...' % embeddings.shape[0])
  output_file_path = os.path.join(out_dir, f'{model_name}_clusters.json')
  if os.path.exists(output_file_path):
      raise ValueError('Cannot overwrite the output dataset!')
  os.makedirs(out_dir, mode = 0o777, exist_ok = True) 
  cluster_and_store(
    embeddings=embeddings, 
    wordpieces=raw_wordpieces,
    knns=knns,
    output_file_path=output_file_path,
  )


def get_glove(vocab_list = None, knns = [200, 125, 100, 75, 50, 25, 12, 6]):
    """Get Glove clusters. 
    
    Note: to get the values reported in the paper, you need to get the subset of Albert adn Glove tokens.
    """
    vec = GloVe()
    # vocab = vocab(vec.stoi)
    glove_vocab = list(vec.stoi.keys())
    # len(glove_vocab)
    embeddings = torch.nn.Embedding.from_pretrained(vec.vectors,freeze=True)
    if vocab_list:
      vocab_list = list(set(glove_vocab).intersection(vocab_list))
      # Note: to get the values reported in the paper, you need to get the subset of Albert adn Glove tokens.
      subset_embedding = torch.nn.Embedding(len(vocab_list),  embeddings.weight.shape[1])
      subset_embedding.weight = torch.nn.Parameter(vec.get_vecs_by_tokens(vocab_list))
    
    print('Running the hierarchical cluster for %s tokens for Albert Model...' % embeddings.shape[0])
    cluster_and_store(
      embeddings=embeddings if not vocab_list else subset_embedding, 
      wordpieces=glove_vocab  if not vocab_list else vocab_list, 
      knns=knns,
      output_file_path='glove_clusters.json',
    )


def main():
  args = parser.parse_args()
  if 'gemma' in args.model.lower():
    # TODO: change it work word for separate binary.
    notebook_login()
    get_Gemma(args.model.lower(), args.output_dir)
  elif 'albert' in args.model.lower():
    get_albert(args.model.lower(), args.output_dir)
  elif 't5' in args.model.lower():
    get_t5(args.model.lower(), args.output_dir)
  elif 'mt5' in args.model.lower():
    get_mt5(args.model.lower(), args.output_dir)
  elif args.model.lower() == 'glove':
    vocab_list = []
    # # This is to get the clusters for the intersection of glove and ALBERT.
    # tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    # raw_wordpieces = tokenizer.sp_model.id_to_piece(list(range(0, 29999)))
    # vocab_list = list(set([tk.replace('▁', '') for tk in raw_wordpieces]))
    # get_glove()
  else:
    raise ValueError('Unsupported Model Name. Pick from [Albert, T5, mT5, Gemma]')


if __name__ == "__main__":
    main()

