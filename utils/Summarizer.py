from summarizer import Summarizer
import torch
import streamlit as st
# torch.load(, map_location="cpu")

def file_text(file):
        text = open('{}'.format(file.name), 'r')
        text = text.read()
        return text

def summary_(text, num_sent): 
        bert_model = Summarizer()
        bert_summary = bert_model(text, num_sentences=num_sent)
        result = bert_summary
        return result


