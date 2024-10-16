import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import ast
import argparse
from huggingface_hub import HfApi, login

def load_tokens(file_path):
    with open(file_path, 'r') as file:
        file_contents = file.read().strip()
    return ast.literal_eval(file_contents)

def get_embedding(token, tokenizer, model):
    token_id = tokenizer.convert_tokens_to_ids(token)
    embedding = model.shared.weight[token_id, :]
    return embedding

def calculate_average_norm(embeddings):
    norms = torch.norm(embeddings, dim=1)
    return torch.mean(norms)

def scale_to_magnitude(average_embedding, U):
    average_embedding_norm = torch.norm(average_embedding)
    scaled_embedding = average_embedding * (U / average_embedding_norm)
    return scaled_embedding

def assign_generated_embeddings(tokens, tokenizer, model):
    embeddings = torch.stack([get_embedding(token, tokenizer, model) for token in tokens])
    U = calculate_average_norm(embeddings)
    mu = torch.mean(embeddings, dim=0)
    mu = scale_to_magnitude(mu, U)
    std = torch.std(embeddings, dim=0)
    gaussian_distribution = torch.distributions.Normal(0, std*0.7)  
    S_i = gaussian_distribution.sample()
    aux = mu + S_i
    aux_norm = torch.norm(aux)
    final_embedding = aux * (U / aux_norm)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    with torch.no_grad():
        for i, token_id in enumerate(ids):
            model.shared.weight[token_id, :] = final_embedding
    return final_embedding

def make_tensors_contiguous(model):
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

def main(token_file_path, model_name, output_model_path, huggingface_token, repo_id):
    # Load the T5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load tokens
    tokens_1 = load_tokens(token_file_path)
    print(f"Loaded {len(tokens_1)} tokens")

    # Generate and assign new embeddings to the tokens
    final_embedding = assign_generated_embeddings(tokens_1, tokenizer, model)
    make_tensors_contiguous(model)

    # Save the updated model and tokenizer
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)
    print(f"Updated model saved to {output_model_path}")

    # Upload to Hugging Face Hub
    login(token=huggingface_token)
    api = HfApi()
    repo_url = api.create_repo(repo_id=repo_id)
    api.upload_folder(
        folder_path=output_model_path,
        repo_id=repo_id,
        repo_type="model"
    )
    print(f"Updated model uploaded successfully to {repo_url}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update T5 model embeddings and upload to Hugging Face Hub")
    parser.add_argument("token_file_path", help="Path to the file containing tokens")
    parser.add_argument("model_name", help="Name or path of the pre-trained model")
    parser.add_argument("output_model_path", help="Path to save the updated model")
    parser.add_argument("huggingface_token", help="Hugging Face API token")
    parser.add_argument("repo_id", help="Repository ID for Hugging Face Hub")
    args = parser.parse_args()

    main(args.token_file_path, args.model_name, args.output_model_path, args.huggingface_token, args.repo_id)
