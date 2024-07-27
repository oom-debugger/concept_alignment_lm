""" API invocatoins to extract clusters for ALBERT, T5, and GloVE."""
import copy
import torch.nn
from torchtext.vocab import GloVe, vocab

from transformers import T5Tokenizer, T5EncoderModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import MT5Model, T5Tokenizer
from huggingface_hub import notebook_login
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model", dest = "model", default = "Albert", help="model name")

def get_Gemma(knns = [125, 100, 75, 50, 25, 12, 6]):
  """get Gemma clusters. You need to login and authenticate your account on HuggingFace."""
  tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
  raw_wordpieces = []
  for i in range(len(tokenizer.vocab.values())):
    raw_wordpieces.append(tokenizer._convert_id_to_token(i))
  model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
  gemma_embeddings = copy.deepcopy(model._modules['model']._modules['embed_tokens'].weight.detach().cpu())
  print('Running the hierarchical cluster for %s tokens for Gemma Model...' % gemma_embeddings.shape[0])
  cluster_and_store(
    embeddings=gemma_embeddings, 
    wordpieces=raw_wordpieces,
    knns=knns,
    output_file_name='gemma_clusters.json',
  )

def get_t5(knns = [125, 100, 75, 50, 25, 12, 6]):
  """get T5 clusters. You need to login and authenticate your account on HuggingFace."""

  tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
  raw_wordpieces = t5_tokenizer.sp_model.id_to_piece(list(range(32000)))
  # raw_wordpieces = [t.lower() for t in raw_wordpieces]
  t5_model = T5EncoderModel.from_pretrained("google-t5/t5-small")
  embeddings = t5_model._modules['shared'].weight.detach().cpu()
  print('Running the hierarchical cluster for %s tokens for T5 Model...' % embeddings.shape[0])
  cluster_and_store(
    embeddings=embeddings, 
    wordpieces=raw_wordpieces,
    knns=knns,
    output_file_name='t5_clusters.json',
  )

def get_mt5(knns = [125, 100, 75, 50, 25, 12, 6]):
  """get T5 clusters. You need to login and authenticate your account on HuggingFace."""

  tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
  raw_wordpieces = tokenizer.sp_model.id_to_piece(list(range(32000)))
  model = MT5Model.from_pretrained("google/mt5-small")
  embeddings = model._modules['shared'].weight.detach().cpu()
  
  print('Running the hierarchical cluster for %s tokens for mT5 Model...' % embeddings.shape[0])
  cluster_and_store(
    embeddings=embeddings, 
    wordpieces=raw_wordpieces,
    knns=knns,
    output_file_name='t5_clusters.json',
  )


def get_albert(knns = [125, 100, 75, 50, 25, 12, 6]):
  """get ALBERT clusters. You need to login and authenticate your account on HuggingFace."""
  tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')  # ALBERT-xxlarge
  raw_wordpieces = tokenizer.sp_model.id_to_piece(list(range(0, 30000)))
  model = AlbertModel.from_pretrained("albert-xxlarge-v2")
  embeddings = model._modules['embeddings']._modules['word_embeddings'].weight.detach().cpu()
  print('Running the hierarchical cluster for %s tokens for Albert Model...' % embeddings.shape[0])
  cluster_and_store(
    embeddings=embeddings, 
    wordpieces=raw_wordpieces,
    knns=knns,
    output_file_name='albert_xxl_clusters.json',
  )

  def get_glove(knns = [200, 125, 100, 75, 50, 25, 12, 6], vocab_list = None):
    """Get Glove clusters. 
    
    Note: to get the values reported in the paper, you need to get the subset of Albert adn Glove tokens.
    """
    vec = GloVe()
    # vocab = vocab(vec.stoi)
    glove_vocab = list(myvec.stoi.keys())
    # len(glove_vocab)
    embeddings = torch.nn.Embedding.from_pretrained(myvec.vectors,freeze=True)
    if vocab_list:
      # Note: to get the values reported in the paper, you need to get the subset of Albert adn Glove tokens.
      subset_embedding = torch.nn.Embedding(len(vocab_list),  embeddings.weight.shape[1])
      subset_embedding.weight = torch.nn.Parameter(myvec.get_vecs_by_tokens(vocab_list))
    
    print('Running the hierarchical cluster for %s tokens for Albert Model...' % embeddings.shape[0])
    cluster_and_store(
      embeddings=embeddings if not vocab_list else subset_embedding, 
      wordpieces=glove_vocab  if not vocab_list else vocab_list, 
      knns=knns,
      output_file_name='albert_xxl_clusters.json',
    )


def main():
  args = parser.parse_args()
  if args.model.lower() == 'gemma':
    # TODO: change it work word for separate binary.
    notebook_login()
    get_Gemma()
  elif args.model.lower() == 'albert':
    get_albert()
  elif args.model.lower() == 't5':
    get_t5()
  elif args.model.lower() == 'mt5':
    get_mt5()
  else:
    raise ValueError('Unsupported Model Name. Pick from [Albert, T5, mT5, Gemma]')


if __name__ == "__main__":
    main()

