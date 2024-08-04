
import copy
import torch
import json
import os.path
import pprint
import warnings
from functools import partial
from sklearn.metrics import classification_report

from datasets import load_dataset, concatenate_datasets, shuffle
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model", dest = "model", default = "Albert", help="huggingface model name or local path.")
parser.add_argument("-base_dir", "--base_dir", dest = "base_dir", default = None, help="Base directory to save artifacts.")


def get_glue_datasets():
  tasks = ['ax', 'cola', 'mnli', 'mnli_matched', 'mnli_mismatched', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
  train_datasets = {}
  validation_datasets = {}   # train eval:
  test_datasets = {}   # test eval:
  for task in tasks:
    try:
      if task != 'mnli':
        train = load_dataset("nyu-mll/glue", task, split="train")
        validation = load_dataset("nyu-mll/glue", task, split="validation")
        test = load_dataset("nyu-mll/glue", task, split="test")
        train_datasets[task] = train
        validation_datasets[task] = validation
        test_datasets[task] = test
      else:
        ds = load_dataset("nyu-mll/glue", task)
        train_datasets['mnli'] = ds['train']
        test_datasets['mnli'] = concatenate_datasets([ds['test_matched'], ds['test_mismatched']]).shuffle(seed=42)
        validation_datasets['mnli'] = concatenate_datasets([ds['validation_matched'], ds['validation_mismatched']]).shuffle(seed=42)
      print(f'successfully read train and test splits for {task}.')
    except:
      pass
  return {
      'train_sets': train_datasets,
      'val_sets': validation_datasets,
      'test_sets': test_datasets,
  }


def get_label(task, label):
  label = int(label)
  label_look_up = {
    'cola': {0: 'unacceptable', 1: 'acceptable'},
    'mnli': {0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'mnli_matched': {0: 'entailment', 1:  'neutral', 2: 'contradiction'},
    'mnli_mismatched': {0:  'entailment' , 1:  'neutral' , 2:  'contradiction' },
    'mrpc': {0:  'not_equivalent', 1:  'equivalent' },
    'qnli': {0:  'entailment', 1:  'not_entailment' },
    'qqp': {0:  'not_duplicate', 1:  'duplicate' },
    'rte': {0:  'entailment', 1:  'not_entailment' },
    'sst2': {0:  'negative' , 1:  'positive' },
    'wnli': {0:  'not_entailment' , 1: 'entailment'},
    'stsb': {0:  'not similar' , 1: 'slightly similar' , 2: 'somewhat similar' , 3: 'moderately similar', 4: 'very similar' , 5: 'extremely similar'},
  }
  if task not in label_look_up:
    raise ValueError('Unsupported task: , existing tasks:', task, label_look_up.keys())
  if label not in label_look_up[task]:
    raise ValueError('Unsupported label: , existing labels:', label, label_look_up[task].keys())
  return label_look_up[task][label]


# entailment (0), neutral (1), contradiction (2)
def formatting_prompts(
    examples,
    task,
    instruction_template, # = ' ### input:',
    response_template, # = ' ### Label:',
    is_inference = False,  # to enable using as part of map function.
    ):
    output_texts = []
    input_keys = list(examples.keys())
    input_keys.remove('label')
    input_keys.remove('idx')
    batch_size = len(examples['label'])
    for i in range(batch_size):
      input_text = '** '.join([f'{k}: {examples[k][i]} ' for k in input_keys])
      if not is_inference:
        text = f"{instruction_template} {input_text}\n {response_template} {get_label(task, examples['label'][i])}"
      else:
        # text = f"{instruction_template} {input_text}\n {response_template} {get_label(task, examples['label'][i])}"
        text = f"{instruction_template} {input_text}\n {response_template} "
      output_texts.append(text)
    return {'text': output_texts} if is_inference else output_texts


def inference_and_eval(
    model,
    tokenizer,
    task_name,
    eval_dataset,
    formatting_prompts_fnc, 
    eval = True):
      """Run the classification eval on the text2text model."""
      inference_model = model.to(device=f'cuda:{torch.cuda.current_device()}')
      inference_tokenizer = tokenizer
      # small_dataset = eval_dataset[task_name].select(range(100))
      updated_dataset = eval_dataset.map(formatting_prompts_fnc, batched=True, remove_columns=list(eval_dataset.features.keys()))
      inputs = inference_tokenizer(updated_dataset['text'], return_tensors="pt", padding=True)
      inputs["input_ids"] = inputs["input_ids"].to(device=f'cuda:{torch.cuda.current_device()}')
      inputs["attention_mask"] = inputs["attention_mask"].to(device=f'cuda:{torch.cuda.current_device()}')
      output_sequences = inference_model.generate(
          input_ids=inputs["input_ids"],
          attention_mask=inputs["attention_mask"],
          do_sample=False,  # disable sampling to test if batching affects output
      )
      pred = inference_tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
      eval_report = None
      if eval:
        truth = [get_label(task_name, label) for label in eval_dataset['label']]
        # convert the labels to avoid having lots of garbage labels (since it is a generative model).
        target_names = list(set(truth))
        target_names.append('other')
        pred = ['other' if p not in target_names else p for p in pred]
        print('binary accuracy is: ', sum([int(p == t) for p, t in zip(pred, truth)]) / len(pred))
        eval_report = classification_report(truth, pred, target_names=target_names, output_dict=True)
      return pred, eval_report


def train_and_eval_glue(
  model_name,
  sft_dict,
  base_dir,
  train_dataset,
  eval_dataset,
  task_name,
  instruction_template,
  response_template):
    """Train and evaluate the Glue benchmark, and store the results and model in the base_dir."""
    eval_file = os.path.join(base_dir, f"{model_name}_{task_name}_eval.json")
    if os.path.isfile(eval_file):
      print(f"Skipping {eval_file} since it already exists.")
      continue 
  
    formatting_prompts_func = partial(
        formatting_prompts,
        task = task_name,
        instruction_template = instruction_template,
        response_template = response_template,
    )
    # 2. train for each task
    if isinstance(model_name,str):
      if 't5' in model_name: # encoder-decoder models, e.g. "google-t5/t5-small"
        model = T5ForConditionalGeneration.from_pretrained(model_name)
      else:  # decoder only, e.g. Gemma
        model = AutoModelForCausalLM.from_pretrained(model_name)
      tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
      model = model_name[0]
      tokenizer = model_name[1]
    # 2.2. data collator for text generation.
    collator = DataCollatorForCompletionOnlyLM(
        # instruction_template=instruction_template,
        response_template=response_template, 
        tokenizer=tokenizer)
    # 2.3. defining SFT trainer.
    # training_args = TrainingArguments(
    sft_dict.update(dict(output_dir=f"{model_name}_{task_name}_checkpoints/"))
    training_args = SFTConfig(**sft_dict)

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        args=training_args,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        # optimizers=(optim, scheduler),
        packing=False if collator else True,  # packing=False Can make training 5x faster for short sequences.
    )
    # 2.4 train and save the model.
    trainer.train()
    model.save_pretrained(os.path.join(base_dir, f"{model_name}_{task_name}"))
    tokenizer.save_pretrained(os.path.join(base_dir,f"{model_name}_{task_name}"))
  
    # 3. evaluate
    formatting_prompts_func_inf = partial(
        formatting_prompts,
        task = task_name,
        instruction_template = instruction_template,
        response_template = response_template,
        is_inference=True
    )
    eval_dataset=eval_dataset
    predictions, eval_report = inference_and_eval(
        model, tokenizer, 
        task_name, eval_dataset, 
        formatting_prompts_func_inf,
        eval=True)
    
    # 4. Write  the eval numbers, (the model and tokenizer is already saved).
    formatting_prompts_func_inf = partial(
        formatting_prompts,
        task = task_name,
        instruction_template = instruction_template,
        response_template = response_template,
        is_inference = True
    )
    eval_dataset= glue_datasets['val_sets'][task_name]
    predictions, eval_report = inference_and_eval(
        model, tokenizer, 
        task_name, eval_dataset, 
        formatting_prompts_func_inf, eval=True)
    with open(eval_file, 'w') as f:
      json.dump(eval_report, f)
    # pprint.pp(eval_report)


def main():
  args = parser.parse_args()
  # TODO: parse yaml SFT config
  sft_config = {
    batch_size = 8
    num_train_epochs = 3.0
    # report_to="wandb",  # this tells the Trainer to log the metrics to W&B
    save_strategy='epoch',
    optim = "adamw_torch",
    adam_epsilon=1e-8,
    # lr_scheduler_type options: [linear_with_warmup, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup, inverse_sqrt]
    lr_scheduler_type="linear", 
    bf16=True,
    learning_rate=1e-5,
    warmup_ratio = 0.1,
  }
  model_name = args.model #"google-t5/t5-small"
  # 1. Load Raw Datasets
  glue_datasets = get_glue_datasets()
  for task_name in glue_datasets['train_sets'].keys():
    train_and_eval_glue(
      model_name,
      sft_dict=sft_config,
      base_dir=args.base_dir,
      train_dataset=glue_datasets['train_sets'][task_name],
      eval_dataset=glue_datasets['test_sets'][task_name],
      task_name=task_name,
      instruction_template=' ### Question:',
      response_template=' ### Answer:')


if __name__ == "__main__":
    main()
