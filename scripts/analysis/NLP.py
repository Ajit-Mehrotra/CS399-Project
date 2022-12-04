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


df = pd.read_csv("../../../Documents/Github/CS399_Project/data/glassdoor_reviews.csv")


columns = ["pros", "cons", "headline"]
df = df.dropna(subset=['work_life_balance'])


#types_str = df.select_dtypes(include='object').columns
df[columns] = df[columns].fillna("NA")


df[columns] = df[columns].astype(str)


# Preprocessing text routines 
stemmer = WordNetLemmatizer()
tok = TreebankWordTokenizer()

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english')) # common english stop words
to_be_removed = list(en_stop) + list(punctuation)

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



# Run me 
def tokenize_pad_numericalize(entry, vocab_stoi, tok, pad=True, max_length=100):
    if pad :
        text = [ vocab_stoi[token] if token in vocab_stoi else vocab_stoi[''] for token in tok.tokenize(entry)]
        padded_text = None
        l = len(text)
        if l < max_length:   padded_text = text + [ vocab_stoi[''] for i in range(len(text), max_length) ] 
        elif l > max_length: padded_text = text[:max_length]
        else:                        padded_text = text
        return padded_text
    else : 
        text = [ vocab_stoi[token] if token in vocab_stoi else vocab_stoi[''] for token in tok.tokenize(entry)]
        return text

# this part gets ran to get the words that are commonly seen together 
pretrained_vectors = FastText(language='en')
pretrained_vocab = vocab(pretrained_vectors.stoi, min_freq=0)

unk_token = ""
unk_index = 0
pad_token = ''
pad_index = 1
pretrained_vocab.insert_token("",unk_index)
#pretrained_vocab.insert_token("",pad_index)

pretrained_vocab.set_default_index(unk_index)
pretrained_embeddings = pretrained_vectors.vectors


pretrained_embeddings = torch.cat((torch.zeros(2,pretrained_embeddings.shape[1]),pretrained_embeddings))
stoi = pretrained_vocab.get_stoi()
tok = TreebankWordTokenizer()

all_sentences = df[['pros','cons']].applymap(lambda x: nltk.word_tokenize(preprocess_transformers(x, full_process=False)))
all_sentences = all_sentences.melt().value.to_list()

phrases = Phrases(all_sentences, delimiter='oo' ,threshold=100)
phraser = Phraser(phrases)


test = False
if test:
    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
else:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Data maker for the transfomer
def make_inputs(df,feature):
    minutes_transformers = df[['date_review',feature]].set_index('date_review').copy()
    # standardize text
    minutes_transformers['sentence_treated'] = minutes_transformers[feature].apply(lambda x: preprocess_transformers(x, full_process=False))
    # Apply phraser
    minutes_transformers['sentence_treated'] = minutes_transformers['sentence_treated'].apply(lambda x: ' '.join(phraser[nltk.word_tokenize(x)]))
    # Set max length
    max_len = minutes_transformers['sentence_treated'].apply(lambda x:len(nltk.word_tokenize(x))).quantile(0.95)
    max_len = np.min((max_len,200)).astype(int)

    encoded_corpus = tokenizer(text=minutes_transformers['sentence_treated'].tolist(),
                            add_special_tokens=True,
                            padding='max_length',
                            truncation='longest_first',
                            max_length=max_len,
                            return_attention_mask=True)
                            
    print('The maximum sentence length for the transformers input is', max_len)
    input_ids      = encoded_corpus['input_ids']
    attention_mask = encoded_corpus['attention_mask']
    return input_ids, attention_mask

# pros and cons to feed to the transformer
input_pros, mask_pros = make_inputs(df,'pros')
input_cons, mask_cons = make_inputs(df,'cons')


inputs = input_pros, input_cons, mask_pros, mask_cons


labels = df.work_life_balance.astype(int).to_numpy() - 1

# Run me
#We use the same function for transformers and for LSTMs
df['headline'] = df['headline'].progress_apply(lambda x : preprocess_transformers(x, full_process=False))
df['pros'] = df['pros'].progress_apply(lambda x : preprocess_transformers(x, full_process=False))
df['cons'] = df['cons'].progress_apply(lambda x : preprocess_transformers(x, full_process=False))

df['headline'] = df['headline'].progress_apply(lambda x: ' '.join(phraser[word_tokenize(x)]))
df['pros'] = df['pros'].progress_apply(lambda x: ' '.join(phraser[word_tokenize(x)]))
df['cons'] = df['cons'].progress_apply(lambda x: ' '.join(phraser[word_tokenize(x)]))


# Run me
pad = True
# We use different lengths for each field, because the headlines are usually much shorter than the pros and cons
df['tokenized_headline'] = df['headline'].progress_apply(lambda x : tokenize_pad_numericalize(x, stoi, tok, pad, max_length=20))
df['tokenized_pros'] = df['pros'].progress_apply(lambda x : tokenize_pad_numericalize(x, stoi, tok, pad, max_length=80))
df['tokenized_cons'] = df['cons'].progress_apply(lambda x : tokenize_pad_numericalize(x, stoi, tok, pad, max_length=100))


pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.options.display.max_colwidth = 300



torch.Tensor(short.iloc[1, -4]).int()


tokenizer.decode(torch.Tensor(short.iloc[1, -4]).int())


def imtrying(row1, row2):
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


s = "very friendly and welcoming to new staff easy going ethic"
s2 = [226, 3378, 8, 17583, 12, 47, 1154, 2204, 667, 24375, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


imtrying(s, s2)


df['tokenized_headline']=df.apply(lambda row: imtrying(row['headline'], row['tokenized_headline']), axis=1)
df['tokenized_pros']=df.apply(lambda row: imtrying(row['pros'], row['tokenized_pros']), axis=1)
df['tokenized_cons']=df.apply(lambda row: imtrying(row['cons'], row['tokenized_cons']), axis=1)


df.drop(['pros','cons','headline'], axis=1, inplace=True)


df.to_csv("tokenized_data2", sep='\t', encoding='utf-8')





