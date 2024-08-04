## Clustering:
> python3 extract_communities.py --model=albert  <br>

### required packages
!pip install -q networkx  <br>
!pip install -q transformers  <br>
!pip install -q umap-learn  <br>
!pip install -q sentencepiece  <br>
!pip install -q torch  <br>
!pip install -q torchtext  <br>

## SFT Training:
### Command:
> python3 text2text_glue_train.py \
   --model=google-t5/t5-small  \
   --base_dir=model_dir/glue/ \
   --baseline_config=/home/mehrdad/git/concept_alignment_lm/configs/glue_baseline.yaml  <br>

### required packages
!pip install -q transformers  <br>
!pip install -q sentencepiece  <br>
!pip install -q datasets  <br>
!pip install -q trl  <br>
