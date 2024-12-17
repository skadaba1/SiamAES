from datasets import load_dataset
from utils import load_and_filter, tokenize
from collections import defaultdict
import itertools

def gen_data(prompt, num_classes, n_each, n_augment, flag, split):
  counts = defaultdict(int)
  dataset, num_classes = load_and_filter(prompt, num_classes, n_each, n_augment, split)
  batch_size = 1
  train_dataloader, eval_dataloader, test_dataloader, essays = tokenize(dataset, batch_size)

  batches_train = []
  for batch_i, batch in enumerate(train_dataloader):
    batches_train.append(batch)
  batches_eval = []
  for batch_i, batch in enumerate(eval_dataloader):
    batches_eval.append(batch)
  batches_test = []
  for batch_i, batch in enumerate(test_dataloader):
    batches_test.append(batch)

  if(flag == 1):
    train_dataloader = list(itertools.combinations(batches_train, 2)); 
    #eval_dataloader = list(itertools.combinations(batches_eval, 2)); 

  elif(flag == 2):

    train_dataloader = list(itertools.permutations(batches_train, 3))
    train_dataloader = [(x,y,z) for (x,y,z) in train_dataloader if x['labels'] == y['labels'] and x['labels'] != z['labels']]

  for batch in batches_test:
    counts[batch['labels'].item()]+=1

  return num_classes, counts, train_dataloader, eval_dataloader, test_dataloader, essays, batches_train, batches_test