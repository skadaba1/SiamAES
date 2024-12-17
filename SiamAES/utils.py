import re
import numpy as np
import pandas as pd
import nltk
from datasets import load_dataset
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from random import randint
from collections import defaultdict
import itertools
import random
import nltk
import torch
nltk.download('all')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from datasets import DatasetDict

from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
import torch.nn as nn
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#--------------------------------------------------------------------------------------------
def remove_punctuation(text):
     punctuations = '''!()-[]{};@:'",./?@#$%^+&*_~'''
     no_punct = ""
     for char in text:
        if char not in punctuations:
            no_punct = no_punct + char
     return no_punct
#--------------------------------------------------------------------------------------------
def preprocess_text(text):
    text = re.sub(r'd+' , '', text)
    text = remove_punctuation(text)
    text = text.lower()
    text = text.strip()
    text = re.sub(r'bw{1,2}b', '', text)
    return text

def normalize_scores(dataset, num_classes, prompt):
  if(num_classes != None):
    s_qcut, percentiles = pd.qcut(dataset[prompt][:]['score'], q=num_classes, duplicates='drop', retbins=True)
    num_classes = len(percentiles)

    def normalize(example):
      flag = False
      for index in range(len(percentiles)-1):
        if(example["score"] >= percentiles[index] and example["score"] < percentiles[index+1]):
          example["score"] = index
          flag=True
      if(not flag):
        example["score"] = num_classes - 1
      example['essay'] = preprocess_text(example['essay'])
      return example

    dataset = dataset.map(normalize)
  else:
    num_classes = max(dataset[prompt][:]["score"])
  return dataset, num_classes

def select_samples(dataset, prompt, num_classes, neach):
  map_indexes = defaultdict(list)
  for i in range(len(dataset[prompt])):
    score = dataset[prompt][i]['score']
    map_indexes[score].append(i)
  return map_indexes

def modify(text, threshold=0.90):
  words = nltk.word_tokenize(text)
  tagged = nltk.pos_tag(words)
  output = ""
  ps = PorterStemmer()
  map={'NNS':'n', 'NNPS':'n', 'NNP':'n', 'NN':'n', 'JJ':'a', 'JJR':'a', 
      'JJS':'a', 'RB':'r', 'RBR':'r', 'RBS':'r', 'WP':'r',
      'VB':'v', 'VBD':'v', 'VBG':'v', 'VBN':'v', 'VBP':'v', 'VBZ':'v'} 
  sample_rate = 1
  for i in range(0,len(words)):
      replacements = []
      epsilon = abs(np.random.randn())
      sample_rate*=threshold
      for syn in wn.synsets(words[i]):
          if tagged[i][1] == 'NNP' or tagged[i][1] == 'NNS' or tagged[i][1] == "NN" or tagged[i][1] == 'DT':
              break          
          word_type = tagged[i][1][0].lower(); 
          if syn.name().find("."+word_type+"."):
              if(tagged[i][1] in map.keys()):
                r = WordNetLemmatizer().lemmatize(syn.name()[0:syn.name().find(".")], map[tagged[i][1]])
              else:
                r = words[i]
              replacements.append(r)
      if len(replacements) > 0:
          replacement = replacements[randint(0,len(replacements)-1)] if epsilon < sample_rate else words[i]
          output = output + " " + replacement
      else:
          output = output + " " + words[i]
  return output

def augment(dataset, prompt, indexes, n_augment):
  last = len(dataset[prompt])
  for i in range(len(indexes)):
    for j in range(n_augment):
      new_item = {'essay':modify(dataset[prompt][indexes[i]]['essay']), 'score':dataset[prompt][indexes[i]]['score']}
      dataset[prompt] = dataset[prompt].add_item(new_item)
      indexes.append(last)
      last += 1
  return dataset, indexes

def train_test_split(l1, l2):
  ind = [0] + list(itertools.accumulate(l2))
  return  [l1[ind[i]:ind[i+1]] for i in range(len(ind)-1)]

def load_and_filter(prompt, num_classes=None, neach=2, n_augment=10, split=[10, 10]):
  dataset = load_dataset("Ericwang/ASAP")
  num_classes = num_classes if num_classes != None else None
  dataset, num_classes = normalize_scores(dataset, num_classes, prompt)
  map_indexes = select_samples(dataset, prompt, num_classes, neach)

  desired = [neach, split[0], split[1]]
  train = []
  val = []
  test = []
  for score in range(num_classes):
    splits = train_test_split(map_indexes[score], desired)
    train += splits[0]
    val += splits[1]
    test += splits[2]

  dataset, indexes = augment(dataset, prompt, train, n_augment)
  random.shuffle(train); random.shuffle(val); random.shuffle(test)
  dataset = DatasetDict(
      train=dataset[prompt].select(indexes),
      val=dataset[prompt].select(val),
      test=dataset[prompt].select(test),
  )
  return dataset, num_classes

def tokenize(dataset, batch_size):
  tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
  tokenized_dataset = dataset.map(
      lambda example: tokenizer(example['essay'], padding='max_length', truncation=True),
      batched=True,
      batch_size=batch_size
  )
  essays = tokenized_dataset['test']['essay']
  tokenized_dataset = tokenized_dataset.remove_columns(["essay"])
  tokenized_dataset = tokenized_dataset.rename_column("score", "labels")
  tokenized_dataset.set_format("torch")

  train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=batch_size, num_workers=4, pin_memory=True);
  eval_dataloader = DataLoader(tokenized_dataset['val'], batch_size=batch_size, num_workers=4, pin_memory=True);
  test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=1)

  return train_dataloader, eval_dataloader, test_dataloader, essays

