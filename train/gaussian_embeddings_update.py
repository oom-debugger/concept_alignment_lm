import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the T5 model and tokenizer. Alternatively, change the tokenizers, models and import appropriate libraies for other models
model_name = ""  
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
import ast

# Specify the path to your file
file_path = 'path/to/your/file.txt'

# Read the contents of the file
with open(file_path, 'r') as file:
    file_contents = file.read().strip()

# Use ast.literal_eval to safely evaluate the string as a Python expression
tokens_1 = ast.literal_eval(file_contents)

# Print the list to verify
print(tokens_1)
# Function to get the embedding of a single token
def get_embedding(token):
    token_id = tokenizer.convert_tokens_to_ids(token)
    embedding = model.shared.weight[token_id, :]
    return embedding

# Calculate the norm for each vector and average the norms (magnitude U)
def calculate_average_norm(embeddings):
    norms = torch.norm(embeddings, dim=1)  # Norm (L2 norm) for each vector
    return torch.mean(norms)  # Average norm

# Scale the average embedding to have magnitude U
def scale_to_magnitude(average_embedding, U):
    average_embedding_norm = torch.norm(average_embedding)
    scaled_embedding = average_embedding * (U / average_embedding_norm)
    return scaled_embedding

# Function to replace the embeddings with the new generated embeddings
def assign_generated_embeddings(tokens):
    # Get original embeddings for all tokens
    embeddings = torch.stack([get_embedding(token) for token in tokens])

    # Calculate U (average of norms)
    U = calculate_average_norm(embeddings)

    # Calculate the average embedding (mu)
    mu = torch.mean(embeddings, dim=0)  # Mean embedding (cluster center)
    mu = scale_to_magnitude(mu, U)  # Scale mean embedding to magnitude U

    # Calculate standard deviation for the embeddings
    std = torch.std(embeddings, dim=0)

    # Generate random samples from normal distribution (S_i)
    gaussian_distribution = torch.distributions.Normal(0, std*0.7)  
    S_i = gaussian_distribution.sample()

    # Compute auxiliary vector (aux = mu + S_i)
    aux = mu + S_i

    # Rescale aux to have the same magnitude as U
    aux_norm = torch.norm(aux)
    final_embedding = aux * (U / aux_norm)

    # Replace original embeddings with new generated embeddings
    ids = tokenizer.convert_tokens_to_ids(tokens)
    with torch.no_grad():
        for i, token_id in enumerate(ids):
            model.shared.weight[token_id, :] = final_embedding

    return final_embedding


def make_tensors_contiguous(model):
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

# Generate and assign new embeddings to the tokens
final_embedding = assign_generated_embeddings(tokens_1)
make_tensors_contiguous(model)
# Save the updated model and tokenizer
model.save_pretrained('')
tokenizer.save_pretrained('')

from huggingface_hub import HfApi, login

# Hugging Face token 
huggingface_token = ""

# Log in using your token
login(token=huggingface_token)

# Initialize the API and create a new repository
api = HfApi()
repo_url = api.create_repo(repo_id="")
api.upload_folder(
    folder_path="",
    repo_id="",
    repo_type="model"
)

print("Updated model uploaded successfully!")

