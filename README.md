## Clustering:
### required packages
!pip install -q networkx  <br>
!pip install -q transformers  <br>
!pip install -q umap-learn  <br>
!pip install -q sentencepiece  <br>
!pip install -q torch  <br>
!pip install -q torchtext  <br>
!pip -q install transformers   <br>
!pip -q install names-dataset   <br>
!pip -q install Unidecode   <br>
!git clone 'https://github.com/dr5hn/countries-states-cities-database.git'   <br>

## commands:
> python3 extract_communities.py --model=albert  <br>

## Evaluation:
### LM-LM alignment:
> python3 interpretability/lm_lm_alignment_embedding.py  \
    --src_model_name=albert-xlarge-v2 \
    --dst_model_name=albert-base-v2  \
    --src_whitespace_char='▁'  \
    --dst_whitespace_char='▁'

### Human-LM alignment:

### Location eval command examples:
> python3 interpretability/location_eval.py \
    --src_cluster=<your dir>/albert-xlarge-v2_clusters.json \
    --src_whitespace_char='▁'  \
    --location_file='<your git dir>/git/countries-states-cities-database/json/countries+states+cities.json' \
    --layer_names=12,25,50,75
> python3 interpretability/location_eval.py \
    --src_cluster=<your dir>/clusters/llama-3.1-70b_clusters.json \
    --src_whitespace_char='Ġ'  \
    --location_file='<your git dir>/git/countries-states-cities-database/json/countries+states+cities.json' \
    --layer_names=6,12,25  \
    --model='meta-llama/Llama-3.1-70B'

### Name eval command examples:
> python3 interpretability/name_eval.py \
    --src_cluster=<your dir>/clusters/llama-3.1-70b_clusters.json \
    --src_whitespace_char='Ġ'  \
    --layer_names=6,12,25  \
    --model='meta-llama/Llama-3.1-70B'

### Gaussian sampling based embeddings
>  python3 gaussian_embeddings_update.py
    --token_file_path /path/to/tokens_from_a_cluster_file.txt \
    --model_name path_to_model \
    --output_model_path /path/to/output_model \
    --huggingface_token your_huggingface_token \
    --repo_id your_repo_id_to_upload_model

## Notes:
HuggingFace Glue dataset, test-split, for some of the tasks the dataset contains invalid labels (-1). Thus we only train set for the training and validation set for the evaluations. <br>
