import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset
from huggingface_hub import HfApi, create_repo
from transformers import TrainerCallback

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Using device: {device}")
    print(f"Number of GPUs available: {num_gpus}")
    return device, num_gpus

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

def prepare_model(model, num_gpus, device):
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    return model.to(device)

class POSDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            words = item["tokens"]
            pos_tags = item["pos_tags"]
            pos_labels = [self.dataset.features["pos_tags"].feature.names[tag] for tag in pos_tags]

            input_text = "pos: " + " ".join(words)
            target_text = " ".join(pos_labels)

            inputs = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
            with self.tokenizer.as_target_tokenizer():
                targets = self.tokenizer(target_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

            input_ids = inputs.input_ids.squeeze()
            attention_mask = inputs.attention_mask.squeeze()
            labels = targets.input_ids.squeeze()
            labels[labels == self.tokenizer.pad_token_id] = -100

            if input_ids.numel() == 0 or attention_mask.numel() == 0 or labels.numel() == 0:
                raise ValueError(f"Empty tensor found for item {idx}")

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None and all(key in item for key in ['input_ids', 'attention_mask', 'labels'])]
    
    if len(batch) == 0:
        print("Warning: Empty batch encountered")
        return None
    
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }

def prepare_datasets(dataset, tokenizer):
    train_dataset = POSDataset(dataset["train"], tokenizer)
    val_dataset = POSDataset(dataset["validation"], tokenizer)
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    return train_dataset, val_dataset

def print_sample_items(dataset, num_samples=3):
    for i in range(num_samples):
        sample = dataset[i]
        if sample is not None:
            print(f"Sample {i}:")
            for key, value in sample.items():
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
        else:
            print(f"Sample {i} is None")

def get_training_arguments(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        fp16=True,
    )

def train_model(model, train_dataset, val_dataset, training_args):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    trainer.train()
    return trainer

def save_model(model, tokenizer, path):
    if isinstance(model, torch.nn.DataParallel):
        model.module.save_pretrained(path)
    else:
        model.save_pretrained(path)
    tokenizer.save_pretrained(path)

def upload_model_to_hf(local_model_path, repo_name, token):
    api = HfApi()
    repo_url = create_repo(repo_name, token=token, private=False, exist_ok=True)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    model.push_to_hub(repo_name, use_auth_token=token)
    tokenizer.push_to_hub(repo_name, use_auth_token=token)

    print(f"Model uploaded successfully to: {repo_url}")
    return repo_url

def main():
    device, num_gpus = setup_device()
    
    dataset = load_dataset("conll2003")
    
    model_name = ""
    model, tokenizer = load_model_and_tokenizer(model_name)
    model = prepare_model(model, num_gpus, device)
    
    train_dataset, val_dataset = prepare_datasets(dataset, tokenizer)
    print_sample_items(train_dataset)
    
    training_args = get_training_arguments("")
    
    trained_model = train_model(model, train_dataset, val_dataset, training_args)
    
    local_model_path = "./gaussian_t11_pos_model_final"
    save_model(trained_model.model, tokenizer, local_model_path)
    
    repo_name = ""
    token = "your_huggingface_token"
    upload_model_to_hf(local_model_path, repo_name, token)

if __name__ == "__main__":
    main()