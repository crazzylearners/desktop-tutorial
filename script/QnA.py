import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import streamlit as st

#text = generator_(input)
@st.cache(allow_output_mutation=True)
def QnA_(ques,text):
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    inputs = tokenizer(ques, text, add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True,)
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer