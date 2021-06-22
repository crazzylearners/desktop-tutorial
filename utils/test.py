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

import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

#text = generator_(input)

def QnA_(ques,text):
    text_inp = text
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    inputs = tokenizer(ques, text_inp, add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True,)
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer
import streamlit as st
text = generator_('machine')
st.title('demo')
q = st.text_input('enter: ')
print(QnA_(q, text))
