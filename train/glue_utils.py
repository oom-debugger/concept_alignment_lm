
import copy
import torch

import datasets as datasets_lib
from functools import partial
from sklearn.metrics import classification_report


def get_glue_datasets():
  tasks = ['ax', 'cola', 'mnli', 'mnli_matched', 'mnli_mismatched', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
  train_datasets = {}
  validation_datasets = {}   # train eval:
  test_datasets = {}   # test eval:
  for task in tasks:
    try:
      if task != 'mnli':
        train = datasets_lib.load_dataset("nyu-mll/glue", task, split="train")
        validation = datasets_lib.load_dataset("nyu-mll/glue", task, split="validation")
        test = datasets_lib.load_dataset("nyu-mll/glue", task, split="test")
        train_datasets[task] = train
        validation_datasets[task] = validation
        test_datasets[task] = test
      else:
        ds = datasets_lib.load_dataset("nyu-mll/glue", task)
        train_datasets['mnli'] = ds['train']
        test_datasets['mnli'] = datasets_lib.concatenate_datasets([ds['test_matched'], ds['test_mismatched']]).shuffle(seed=42)
        validation_datasets['mnli'] = datasets_lib.concatenate_datasets([ds['validation_matched'], ds['validation_mismatched']]).shuffle(seed=42)
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
