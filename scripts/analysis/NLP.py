import pandas as pd
import os
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from termcolor import colored
import torch 
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import FastText, vocab
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from gensim.models.phrases import Phrases, Phraser
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer
from string import punctuation
import re, random, pathlib
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()
import random

from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel


# nltk requirements for words 
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt')

def get_tokenized_data(data: pd.DataFrame) -> pd.DataFrame:  
    # Tokenize the data
    columns = ["pros", "cons", "headline"]
    data = data.dropna(subset=['work_life_balance'])

    data[columns] = data[columns].fillna("NA")
    data[columns] = data[columns].astype(str)

      # Preprocessing text routines 
    stemmer = WordNetLemmatizer()
    tok = TreebankWordTokenizer()

    nltk.download('stopwords')
    en_stop = set(nltk.corpus.stopwords.words('english')) # common english stop words
    to_be_removed = list(en_stop) + list(punctuation)

    # this part gets ran to get the words that are commonly seen together 
    pretrained_vectors = FastText(language='en') # learning of word representation and sentence classification
    pretrained_vocab = vocab(pretrained_vectors.stoi, min_freq=0) # defines vocab object that will be used to numericalize the text/field

    # token objects 
    unk_token = ""
    unk_index = 0
    pad_token = ''
    pad_index = 1
    pretrained_vocab.insert_token("",unk_index)

    pretrained_vocab.set_default_index(unk_index)
    pretrained_embeddings = pretrained_vectors.vectors # get vocab vector for numeric thres. 
    pretrained_embeddings = torch.cat((torch.zeros(2,pretrained_embeddings.shape[1]),pretrained_embeddings))
    vocab_stoi = pretrained_vocab.get_stoi()

    tok = TreebankWordTokenizer()

    all_sentences = data[['pros','cons']].applymap(lambda x: nltk.word_tokenize(preprocess_transformers(x, full_process=False)))
    all_sentences = all_sentences.melt().value.to_list()
    
    # which words commonly co-occur (phrases within our strings)
    phrases = Phrases(all_sentences, delimiter='oo' ,threshold=100)
    phraser = Phraser(phrases)


    test = False
    if test:
        tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    transform_text(data)
    return 

# perform token transformations on the pros, cons, and headline columns
def transform_text(data: pd.DataFrame) -> pd.DataFrame:
    # preprocess the columns in order to tokenize them
    data['headline'] = data['headline'].progress_apply(lambda x : preprocess_transformers(x, full_process=False))
    data['pros'] = data['pros'].progress_apply(lambda x : preprocess_transformers(x, full_process=False))
    data['cons'] = data['cons'].progress_apply(lambda x : preprocess_transformers(x, full_process=False))

    # numeric tokenization
    data['headline'] = data['headline'].progress_apply(lambda x: ' '.join(phraser[word_tokenize(x)]))
    data['pros'] = data['pros'].progress_apply(lambda x: ' '.join(phraser[word_tokenize(x)]))
    data['cons'] = data['cons'].progress_apply(lambda x: ' '.join(phraser[word_tokenize(x)]))

    # padding is used in the numeric tokenization in order to make the tensors the same size (batching)
    pad = True
    # We use different lengths for each field, because the headlines are usually much shorter than the pros and cons
    data['tokenized_headline'] = data['headline'].progress_apply(lambda x : tokenize_pad_numericalize(x, stoi, tok, pad, max_length=20))
    data['tokenized_pros'] = data['pros'].progress_apply(lambda x : tokenize_pad_numericalize(x, stoi, tok, pad, max_length=80))
    data['tokenized_cons'] = data['cons'].progress_apply(lambda x : tokenize_pad_numericalize(x, stoi, tok, pad, max_length=100))

    # Preprocess text for transformers
    data['tokenized_headline']=data.apply(lambda row: str_replace_tokens(row['headline'], row['tokenized_headline']), axis=1)
    data['tokenized_pros']=data.apply(lambda row: str_replace_tokens(row['pros'], row['tokenized_pros']), axis=1)
    data['tokenized_cons']=data.apply(lambda row: str_replace_tokens(row['cons'], row['tokenized_cons']), axis=1)

    # Save data as csv file
    data.drop(['pros','cons','headline'], axis=1, inplace=True)
    data.to_csv("tokenized_data2", sep='\t', encoding='utf-8')


# Preprocess text for transformers
def preprocess_transformers(string, full_process=True):
        # Remove all the special characters
        string = re.sub(r'\W', ' ', str(string))
        # remove all single characters
        string = re.sub(r'\s+[a-zA-Z]\s+', ' ', string)
        # Remove single characters from the start
        string = re.sub(r'\^[a-zA-Z]\s+', ' ', string)
        # Substituting multiple spaces with single space
        string = re.sub(r'\s+', ' ', string, flags=re.I)
        # Removing prefixed 'b'
        string = re.sub(r'^b\s+', '', string)
        # Converting to Lowercase
        string = string.lower()
        if full_process:
            # Lemmatization
            tokens = string.split()
            tokens = [stemmer.lemmatize(word) for word in tokens]
            tokens = [word for word in tokens if word not in en_stop]
            tokens = [word for word in tokens if len(word) > 3]
            string = ' '.join(tokens)
        return string

# numeric tokenization
def tokenize_pad_numericalize(entry, vocab_stoi, tok, pad=True, max_length=100):

    if pad :
        text = [vocab_stoi[token] if token in vocab_stoi else vocab_stoi[''] for token in tok.tokenize(entry)]
        padded_text = None
        l = len(text)
        if l < max_length:   padded_text = text + [ vocab_stoi[''] for i in range(len(text), max_length) ] 
        elif l > max_length: padded_text = text[:max_length]
        else:                        padded_text = text
        return padded_text
    else : 
        text = [ vocab_stoi[token] if token in vocab_stoi else vocab_stoi[''] for token in tok.tokenize(entry)]
        return text

# replace the numerical scores within the tokenized data columns to the corresponding token string 
def str_replace_tokens(row1, row2):
    list_tokens = []
    if len(row1.split()) < len(row2):
        for idx, val in enumerate(row1.split()):
            if row2[idx] > 1000:
                list_tokens.append(val)
    else:
        for idx, val in enumerate(row2):
            if val > 1000:
                list_tokens.append(row1[idx])
    return " ".join(list_tokens)






