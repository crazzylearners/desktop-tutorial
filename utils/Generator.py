import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import re
import tensorflow as tf
import string
import numpy as np
import requests
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

      
def text_generator(txt, url):
  r = requests.get(url)
  soup = BeautifulSoup(r.text, 'html.parser')

  # TEXT
  for para in soup.find_all('p'):
    txt.append(para.text)
  return txt

def link_generator(url):
  r = requests.get(url)
  print(r.status_code)
  soup = BeautifulSoup(r.text, 'html.parser')

  # text
  complete_links = []
  for links in soup.find_all('a'):
    link = links.get('href')
    if link != None:
      if ".org" in link and link[0:5] != 'https':
        complete_links.append("https:"+link)
      elif link[0] != '#'and link[0:5] != 'https':
        complete_links.append("https://en.wikipedia.org"+link)
  return complete_links

def text_cleaning(data):
    tokens = data.split()
    table = str.maketrans('','',string.punctuation)
    tokens = [word.translate(table) for word in tokens]
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return tokens

def generate_text(model, tokenizer, text_seq_length, seed_text, n_words):
  text = []

  for _ in range(n_words):
    encode = tokenizer.texts_to_sequences([seed_text])[0]
    encode = pad_sequences([encode], maxlen = text_seq_length, truncating = 'pre')

    y_pred = model.predict_classes(encode)

    predicted_word = ''
    for word,index in tokenizer.word_index.items():
      if index == y_pred:
        predicted_word = word
        break
    
    seed_text = seed_text + ' ' + predicted_word
    text.append(predicted_word)
  return ' '.join(text)

def generator_(input):
    #text extraction
    txt = []

    context=re.sub(" ",'_',input)

    txt = text_generator(txt,'https://en.wikipedia.org/wiki/'+context)
    links = link_generator('https://en.wikipedia.org/wiki/'+context)

    for i in links[:2]:
        text_generator(txt,i)

    txt = ' '.join(txt)
    text = ' '.join(txt.split())
    text = text.lower()
    return text

def model_(input_text, seed_text, n_words):
    text = ''.join(input_text)
    text = text.split('\n')
    text = ' '.join(text)
    text = ' '.join(text.split())

    clean_text = text_cleaning(text)
    #Lets make dataset for 50 feature
    X = list()
    for i in range(len(clean_text)-51):
        X.append(clean_text[i:i+51])
    # Converting
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    seq = tokenizer.texts_to_sequences(X) 
    seq = np.array(seq)

    # X,y
    X, y = seq[:,:-1], seq[:,-1]
    vocab_size = len(tokenizer.word_index)+1
    y = to_categorical(y,num_classes=vocab_size)
    seq_length = X.shape[1]

    # LSTM
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=seq_length))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100))
    model.add(Dense(vocab_size,activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    model.fit(X,y, batch_size=56, epochs=90)

    return generate_text(model, tokenizer, seq_length, seed_text, n_words)



  
  
