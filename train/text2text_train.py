
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model", dest = "model", default = "Albert", help="huggingface model name or local path.")
parser.add_argument("-base_dir", "--base_dir", dest = "base_dir", default = None, help="Base directory to save artifacts.")

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
